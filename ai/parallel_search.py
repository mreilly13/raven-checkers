from ai.utils import infinity, argmax, argmax_random_tie, num_or_str, Dict, update
from pyspark.sql import types, functions as sf
from pickle import dumps, loads

# naive minimax implementation for comparison
def naive_minimax(state, game, spark, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action.
    This version cuts off search and uses an evaluation function."""
    player = game.to_move(state)

    stats = {"nodes": 0, "depth": d}

    def max_value(st, depth):
        if cutoff_test(st, depth):
            stats["nodes"] += 1
            return eval_fn(st)
        v = -infinity
        successor = game.successors(st)
        for _, s in successor:
            v = max(v, min_value(loads(dumps(s)), depth+1))
        return v

    def min_value(st, depth):
        if cutoff_test(st, depth):
            stats["nodes"] += 1
            return eval_fn(st)
        v = infinity
        successor = game.successors(st)
        for _, s in successor:
            v = min(v, max_value(loads(dumps(s)), depth+1))
        return v

    # Body of alphabeta_search starts here:
    # The default test cuts off at depth d or at a terminal st
    cutoff_test = (cutoff_test or
                   (lambda st, depth: depth > d or game.terminal_test(st)))
    eval_fn = eval_fn or (lambda st: game.utility(player, st))
    action, state = argmax_random_tie(game.successors(state),
                                      lambda a_s: min_value(a_s[1], 0))
    return action, stats

def parallel_minimax(state, game, spark, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action using Spark for parallelism.
    This version cuts off search and uses an evaluation function."""
    
    # deserialize move or state from binary representation stored in DataFrame
    def unpack_eval(bin):
        return eval_fn(loads(bin))

    # The default test cuts off at depth d or at a terminal st
    player = game.to_move(state)
    cutoff_test = (cutoff_test or
                   (lambda st, depth: depth > d or game.terminal_test(st)))
    eval_fn = eval_fn or (lambda st: game.utility(player, st))

    stats = {"nodes": 0, "depth": d}

    # define schema for DataFrame
    schema = sf.StructType([
            types.StructField("move", types.BinaryType(), True),
            types.StructField("state", types.BinaryType(), False)])
    
    # populate DataFrame using successors function to specified depth
    df = spark.createDataFrame(successors(game, d, stats), schema=schema)
    
    # define UDF to evaluate states
    udf_eval = sf.udf(unpack_eval, types.FloatType())
    
    # add new column with evaluated states
    df = df.withColumns({"state_eval": udf_eval(df["state"])})
    
    #compute the best move by selecting the row with the highest evaluation (best for active player, worst for opponent)
    return loads(df.orderBy(sf.desc("state_eval")).first()["move"]), stats

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