"""One execution owns its subprocesses; cancellation always waits for GPU teardown."""
import contextvars
import subprocess
import threading

_CURRENT = contextvars.ContextVar('art_execution', default=None)


class Cancelled(Exception):
    """Cooperative cancellation; not a KeyboardInterrupt, so GUI/asyncio loops do not treat it as Ctrl+C."""


def is_cancellation(error):
    return isinstance(error, (KeyboardInterrupt, Cancelled))


class Session:
    def __init__(self):
        self.cancelled = threading.Event()
        self.active = False
        self._lock = threading.Lock()

    def cancel(self):
        self.cancelled.set()

    def check(self):
        if self.cancelled.is_set():
            raise Cancelled('Processing cancelled')

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Session already running')
        if _CURRENT.get() is not None:
            self._lock.release()
            raise RuntimeError('Nested processing sessions are not supported')
        self.active = True
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


def run_process(args, **kwargs):
    checkpoint()
    with subprocess.Popen(args, **kwargs) as process:
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    checkpoint()
            checkpoint()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, args, stdout, stderr)
            return stdout
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.communicate()  # Do not release the session while inference still owns GPU state.
            raise
