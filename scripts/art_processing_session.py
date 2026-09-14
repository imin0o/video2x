"""One execution owns its subprocesses; cancellation always waits for GPU teardown.

Progress callbacks run synchronously on the processing thread; a GUI must marshal them to its UI thread.
Event keys: run_id, stage, pass_index, pass_count, completed, total, unit, total_exact, elapsed_seconds, state.
A total with total_exact=False is an estimate: reaching it is not completion. Only an event whose state is
completed, failed or cancelled ends a run.
"""
import contextlib
import contextvars
import subprocess
import threading
import time

_CURRENT = contextvars.ContextVar('art_execution', default=None)
_END = object()
STAGES = ('prepare', 'decode-source', 'input-transform', 'inference', 'verify-inference', 'reinput-resize',
          'output-transform', 'encode-output', 'mux-audio', 'verify-output', 'verify-audio')


class Cancelled(Exception):
    """Cooperative cancellation; not a KeyboardInterrupt, so GUI/asyncio loops do not treat it as Ctrl+C."""


def is_cancellation(error):
    return isinstance(error, (KeyboardInterrupt, Cancelled))


class Session:
    """Single use: a cancel requested before start stays effective; a finished session never runs again."""

    def __init__(self, on_progress=None):
        self.cancelled = threading.Event()
        self.active = self.used = False
        self._lock, self._cancel_lock = threading.Lock(), threading.Lock()
        self.on_progress = on_progress
        self.started = self.cancel_requested_at = None
        self.run_id = self.pass_count = self.stage = None
        self.last_progress, self.timings = {}, {}

    def cancel(self):
        with self._cancel_lock:
            if self.cancel_requested_at is None:
                self.cancel_requested_at = time.perf_counter()
            self.cancelled.set()

    def check(self):
        if self.cancelled.is_set():
            raise Cancelled('Processing cancelled')

    def emit(self, stage, pass_index=None, completed=None, total=None, unit=None, total_exact=None, state='running'):
        event = dict(run_id=self.run_id, stage=stage, pass_index=pass_index, pass_count=self.pass_count,
                     completed=completed, total=total, unit=unit, total_exact=total_exact,
                     elapsed_seconds=time.perf_counter() - self.started, state=state)
        self.last_progress = event
        if self.on_progress:
            self.on_progress(event)

    def finish(self, state):
        self.emit(self.stage, state=state)

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Session already running')
        if self.used or _CURRENT.get() is not None:
            self._lock.release()
            raise RuntimeError('Sessions are single-use and cannot be nested; create one per run')
        self.used = self.active = True
        self.started = time.perf_counter()
        self._token = _CURRENT.set(self)
        return self

    def __exit__(self, *unused):
        _CURRENT.reset(self._token)
        self.active = False
        self._lock.release()


def checkpoint():
    session = _CURRENT.get()
    if session:
        session.check()


def progress(stage, completed=None, total=None, pass_index=None, unit='frames', total_exact=True):
    session = _CURRENT.get()
    if session:
        session.check()
        session.stage = stage
        session.emit(stage, pass_index, completed, total, unit if completed is not None else None,
                     total_exact if total is not None else None)


def add_time(operation, started):
    """Accumulate non-overlapping wall time per current stage and operation."""
    session = _CURRENT.get()
    if session:
        key = f'{session.stage or "setup"}/{operation}'
        session.timings[key] = session.timings.get(key, 0.0) + time.perf_counter() - started


@contextlib.contextmanager
def timed(operation):
    started = time.perf_counter()
    try:
        yield
    finally:
        add_time(operation, started)


def timed_iter(iterable, operation):
    """Times only fetching each item, not the consumer's loop body."""
    iterator = iter(iterable)
    while True:
        with timed(operation):
            item = next(iterator, _END)
        if item is _END:
            return
        yield item


def run_process(args, on_tick=None, **kwargs):
    checkpoint()
    with subprocess.Popen(args, **kwargs) as process:
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    checkpoint()
                    if on_tick:
                        on_tick()
            checkpoint()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, args, stdout, stderr)
            return stdout
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.communicate()  # Do not release the session while inference still owns GPU state.
            raise
