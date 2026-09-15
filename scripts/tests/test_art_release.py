"""M5 evidence must fail closed and keep subjective acceptance pending."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from art_model_inspect import sha256
from art_probe_support import save_json
from art_release_report import accept, summarize
from art_release_verify import verify


class ReleaseTests(unittest.TestCase):
    def fixture(self, root):
        for name in ('processing', 'video', 'gui', 'workflow'):
            (root / name).mkdir()
        save_json(root / 'processing/summary.json', dict(status='completed', checks=dict(exact=True)))
        save_json(root / 'video/summary.json', dict(status='completed', checks=dict(exact=True),
                  performance=dict(five_seconds=dict(wall_seconds=1)), cancelled=dict(response_seconds=.1)))
        save_json(root / 'gui/summary.json', dict(status='passed'))
        save_json(root / 'workflow/summary.json', dict(status='passed'))
        save_json(root / 'workflow/create.json', dict(status='passed', A03=True))
        for name in ('a', 'melt-24'):
            directory = root / 'processing' / name
            directory.mkdir()
            video = directory / 'result.mkv'
            video.write_bytes(b'verified output')
            save_json(directory / 'run.json', dict(result=str(video), result_sha256=sha256(video), final_frames=[{}]))

    def test_subjective_approval_not_inferred_from_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            record = summarize(root)
            self.assertEqual(record['status'], 'technical_passed_creator_pending')
            self.assertEqual(record['acceptance']['A07']['creator'], 'pending_flicker_review')
            self.assertEqual(record['acceptance']['A09']['creator'], 'pending_wait_time_limit')
            self.assertEqual(len(record['adopted']), 2)

    def test_false_check_and_modified_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            save_json(root / 'processing/summary.json', dict(status='completed', checks=dict(exact=False)))
            with self.assertRaisesRegex(ValueError, 'did not pass'):
                summarize(root)
            save_json(root / 'processing/summary.json', dict(status='completed', checks=dict(exact=True)))
            (root / 'processing/a/result.mkv').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'output changed'):
                summarize(root)

    def test_creator_decision_is_bound_to_exact_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / 'video/full').mkdir()
            for name in ('environment.json', 'video/full/result.mkv', 'video/full/recipe.json', 'video/full/run.json'):
                (root / name).write_bytes(b'evidence')
            save_json(root / 'verification.json', dict(status='technical_passed_creator_pending',
                                                      originals_and_environment_preserved=True))
            names = ('environment.json', 'verification.json', 'processing/summary.json', 'video/summary.json',
                     'gui/summary.json', 'workflow/create.json', 'workflow/summary.json',
                     'video/full/result.mkv', 'video/full/recipe.json', 'video/full/run.json')
            decision = dict(schema_version=1, status='accepted_by_creator', acceptance=dict(A07=True, A09=True),
                            creator_instruction='Complete R21', date='2026-09-15', verification_directory=str(root),
                            evidence={name: sha256(root / name) for name in names})
            path = root / 'decision.json'
            save_json(path, decision)
            record = summarize(root)
            accept(root, record, path)
            self.assertEqual(record['status'], 'completed_accepted')
            self.assertEqual(record['pending'], [])
            self.assertEqual(record['acceptance']['A07']['creator'], 'accepted_by_creator')
            with self.assertRaisesRegex(ValueError, 'does not accept this verification'):
                accept(root / 'different-run', summarize(root), path)
            (root / 'video/full/result.mkv').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Accepted evidence changed'):
                accept(root, summarize(root), path)

    def test_child_failure_persisted_without_approval_page(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'release'
            with patch('art_release_verify.inventory', return_value={}), \
                    patch('art_release_verify.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'verify')):
                with self.assertRaises(subprocess.CalledProcessError):
                    verify(output, Path('baseline'), Path('cli'))
            self.assertEqual(json.loads((output / 'summary.json').read_text())['status'], 'failed')
            self.assertFalse((output / 'index.html').exists())

    def test_changed_environment_rejects_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'release'
            with patch('art_release_verify.inventory', side_effect=[{'hash': 'a'}, {'hash': 'b'}]), \
                    patch('art_release_verify.subprocess.run'), patch('art_release_verify.write_report') as report:
                with self.assertRaisesRegex(ValueError, 'changed during verification'):
                    verify(output, Path('baseline'), Path('cli'))
                report.assert_not_called()
            self.assertEqual(json.loads((output / 'verification.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
