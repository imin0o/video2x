"""Single-use Session, progress event contract and stage timing."""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from art_processing_session import Cancelled, Session, checkpoint, progress, timed, timed_iter


class SessionTests(unittest.TestCase):
    def test_cancel_before_start_is_effective_and_sessions_are_single_use(self):
        session = Session()
        session.cancel()
        first = session.cancel_requested_at
        session.cancel()
        self.assertEqual(session.cancel_requested_at, first)
        with self.assertRaises(Cancelled), session:
            checkpoint()
        self.assertFalse(session.active)
        with self.assertRaisesRegex(RuntimeError, 'single-use'):
            with session:
                pass
        with Session():
            checkpoint()

    def test_elapsed_time_starts_when_the_run_starts(self):
        events = []
        session = Session(on_progress=events.append)
        time.sleep(.05)
        with session:
            session.run_id, session.pass_count = 'run', 2
            progress('inference', 3, 10, pass_index=2)
            progress('decode-source', 4, 20, total_exact=False)
            progress('mux-audio')
            session.finish('completed')
        self.assertLess(events[0]['elapsed_seconds'], .05)
        self.assertEqual(list(events[0]), ['run_id', 'stage', 'pass_index', 'pass_count', 'completed', 'total',
                                           'unit', 'total_exact', 'elapsed_seconds', 'state'])
        self.assertEqual([(e['stage'], e['pass_index'], e['unit'], e['total_exact'], e['state']) for e in events],
                         [('inference', 2, 'frames', True, 'running'), ('decode-source', None, 'frames', False, 'running'),
                          ('mux-audio', None, None, None, 'running'), ('mux-audio', None, None, None, 'completed')])
        self.assertEqual({e['run_id'] for e in events} | {e['pass_count'] for e in events}, {'run', 2})

    def test_timings_are_keyed_by_stage_and_exclude_loop_bodies(self):
        with Session() as session:
            with timed('file_hash'):
                pass
            progress('encode-output')
            for _ in timed_iter(range(2), 'decode'):
                time.sleep(.02)
        self.assertEqual(set(session.timings), {'setup/file_hash', 'encode-output/decode'})
        self.assertLess(session.timings['encode-output/decode'], .02)


if __name__ == '__main__':
    unittest.main()
