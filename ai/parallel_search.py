import copy
from ai.utils import infinity, argmax, argmax_random_tie, num_or_str, Dict, update
from ai.utils import if_, Struct, abstract
from pyspark.sql import SparkSession, types, functions as sf
from pyspark import RDD
from pickle import dumps, loads

def naive_minimax(state, game, spark, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action.
    This version cuts off search and uses an evaluation function."""
    player = game.to_move(state)
    stats = {"nodes": 0,
             "depth": 0}

    def max_value(st, depth):
        if cutoff_test(st, depth):
            return eval_fn(st)
        v = -infinity
        successor = game.successors(st)
        stats["depth"] = max(stats["depth"], depth)
        for (a, s) in successor:
            v = max(v, min_value(loads(dumps(s)), depth+1))
            stats["nodes"] += 1
        return v

    def min_value(st, depth):
        if cutoff_test(st, depth):
            return eval_fn(st)
        v = infinity
        successor = game.successors(st)
        stats["depth"] = max(stats["depth"], depth)
        for (a, s) in successor:
            v = min(v, max_value(loads(dumps(s)), depth+1))
            stats["nodes"] += 1
        return v

    # Body of alphabeta_search starts here:
    # The default test cuts off at depth d or at a terminal st
    cutoff_test = (cutoff_test or
                   (lambda st, depth: depth > d or game.terminal_test(st)))
    eval_fn = eval_fn or (lambda st: game.utility(player, st))
    action, state = argmax_random_tie(game.successors(state),
                                      lambda a_s: min_value(a_s[1], 0))
    print(f"Naive Minimax: explored {stats['nodes']} nodes; max depth {stats['depth']}")
    return action

def parallel_minimax(state, game, spark, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action using Spark for parallelism.
    This version cuts off search and uses an evaluation function."""
    
    def unpack_eval(bin):
        return eval_fn(loads(bin))

    # Body of search starts here:
    # The default test cuts off at depth d or at a terminal st
    player = game.to_move(state)
    cutoff_test = (cutoff_test or
                   (lambda st, depth: depth > d or game.terminal_test(st)))
    eval_fn = eval_fn or (lambda st: game.utility(player, st))

    stats = {"nodes": 0, "depth": d}
    # sample_space = spark.sparkContext.parallelize(successors(game, d, stats)).toDF(["move", "state"])
    # udf_eval = sf.udf(lambda a, b: (a, eval_fn(b)))

    # processed = sample_space.map(udf_eval)
    # best_row = processed.max(lambda row: row[1])
    # print(f"Parallel Minimax: explored {stats['nodes']} nodes at depth {d}")
    # return best_row[0]

    schema = sf.StructType([
            types.StructField("move", types.BinaryType(), True),
            types.StructField("state", types.BinaryType(), False)])
    df = spark.createDataFrame(successors(game, d, stats), schema=schema)
    udf_eval = sf.udf(unpack_eval, types.FloatType())
    df = df.withColumns({"state_eval": udf_eval(df["state"])}).select("move", "state_eval")
    print(f"Parallel Minimax: explored {stats['nodes']} nodes at depth {stats["depth"]}")
    return loads(df.orderBy(sf.desc("state_eval")).first()["move"])

def successors(game, max_depth, stats, depth=1, move=None):
    state = game.curr_state
    moves = game.legal_moves(state)
    top = move is None
    if not moves:
        return
    else:
        undone = False
        try:
            try:
                for m in moves:
                    if top:
                        move = m
                    undone = False
                    game.make_move(m, state, False)
                    stats["nodes"] += 1
                    if depth < max_depth:
                        yield from successors(game, max_depth, stats, depth + 1, move)
                    else:
                        yield dumps(move), dumps(game.curr_state)
                    game.undo_move(m, state, False)
                    if top:
                        move = None
                    undone = True
            except GeneratorExit:
                raise
        finally:
            if moves and not undone:
                game.undo_move(m, state, False)