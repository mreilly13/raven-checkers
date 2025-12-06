import csv
import ai.games as games
import copy
import multiprocessing
import time
from base.controller import Controller
from util.globalconst import OUTLINE_COLOR, DARK_SQUARES, MAX_DEPTH
import ai.parallel_search as parallel_search
from pyspark.sql import SparkSession

class MinimaxController(Controller):
    def __init__(self, **props):
        self._model = props['model']
        self._view = props['view']
        self._end_turn_event = props['end_turn_event']
        self._highlights = []
        self._search_time = props['searchtime']  # in seconds
        self._before_turn_event = None
        self._parent_conn, self._child_conn = multiprocessing.Pipe()
        self._term_event = multiprocessing.Event()
        self.process = multiprocessing.Process()
        self._start_time = None
        self._call_id = 0
        self._search_algorithm = props['search']
        self._display_metrics = props['display_metrics'] if 'display_metrics' in props else False

    def set_before_turn_event(self, evt):
        self._before_turn_event = evt

    def add_highlights(self):
        for h in self._highlights:
            self._view.highlight_square(h, OUTLINE_COLOR)

    def remove_highlights(self):
        for h in self._highlights:
            self._view.highlight_square(h, DARK_SQUARES)

    def start_turn(self):
        if self._model.terminal_test():
            self._before_turn_event()
            self._model.curr_state.attach(self._view)
            return
        self._view.update_statusbar('Thinking ...')
        self.process = multiprocessing.Process(target=calc_move,
                                               args=(self._model,
                                                     self._search_time,
                                                     self._term_event,
                                                     self._child_conn,
                                                     self._search_algorithm,
                                                     self._display_metrics))
        self._start_time = time.time()
        self.process.daemon = True
        self.process.start()
        self._view.canvas.after(100, self.get_move)

    def get_move(self):
        self._highlights = []
        moved = self._parent_conn.poll()
        while (not moved and (time.time() - self._start_time)
               < self._search_time * 2):
            self._call_id = self._view.canvas.after(500, self.get_move)
            return
        self._view.canvas.after_cancel(self._call_id)
        move = self._parent_conn.recv()
        self._before_turn_event()

        # highlight remaining board squares used in move
        step = 2 if len(move.affected_squares) > 2 else 1
        for m in move.affected_squares[0::step]:
            idx = m[0]
            self._view.highlight_square(idx, OUTLINE_COLOR)
            self._highlights.append(idx)

        self._model.curr_state.attach(self._view)
        self._model.make_move(move, None, True, True,
                              self._view.get_annotation())
        # a new move obliterates any more redo's along a branch of the game tree
        self._model.curr_state.delete_redo_list()
        self._end_turn_event()

    def set_search_time(self, tm):
        self._search_time = tm  # in seconds

    def stop_process(self):
        self._term_event.set()
        if self._call_id != 0:
            self._view.canvas.after_cancel(self._call_id)

    def end_turn(self):
        self._view.update_statusbar()
        self._model.curr_state.detach(self._view)


def longest_of(moves):
    length = -1
    selected = None
    for move in moves:
        current_length = len(move.affected_squares)
        if current_length > length:
            length = current_length
            selected = move
    return selected


def calc_move(model, search_time, term_event, child_conn, search_algorithm, display_metrics=False):
    term_event.clear()
    captures = model.captures_available()
    data = []
    spark_session = SparkSession.builder \
            .appName("MinimaxSearch") \
            .config('spark.driver.memory', '8g') \
            .config('spark.executor.memory', '4g') \
            .config('spark.python.worker.faulthandler.enabled', 'true') \
            .getOrCreate()
    if captures:
        time.sleep(0.7)
        move = longest_of(captures)
    else:
        depth = 0
        start_time = time.time()
        curr_time = start_time
        elapsed_time = 0
        model_copy = copy.deepcopy(model)
        while 1:
            depth += 1
            move, stats = search_algorithm(model_copy.curr_state,
                                          model_copy,
                                          spark_session,
                                          d=depth)
            checkpoint = curr_time
            curr_time = time.time()
            spark_session.catalog.clearCache()
            elapsed_time += curr_time - checkpoint
            data.append(round(elapsed_time, 3))
            if display_metrics:
                print(f"explored {stats['nodes']} nodes in {round(elapsed_time, 3)}s; max depth {stats['depth']}")

            if term_event.is_set():  # a signal means terminate
                term_event.clear()
                move = None
                write_results_to_csv(data)
                break
            if elapsed_time > search_time or depth == 4:
                write_results_to_csv(data)
                break
    spark_session.stop()
    child_conn.send(move)

def write_results_to_csv(results, filename="results.csv"):
    with open(filename, "a") as f:
        csv.writer(f).writerow(results)
