"""Standalone packaging and baseline training-contract regression tests.

Run from CRM-DER: python -B -m unittest discover -s tests -v
Synthetic tests do not measure research performance or read real test labels.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import torch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from opt import parse_opts
from src.backbone.model import BaselineModel
from src.data import load_dataset
from src.crm.factory import resolve_time_unit
from src.trainer import run


def fixture(directory):
    data = Path(directory) / 'data' / 'TINY'
    data.mkdir(parents=True)
    (data / 'entity2id.txt').write_text(''.join('e{}\t{}\n'.format(i, i) for i in range(6)))
    (data / 'relation2id.txt').write_text('r0\t0\nr1\t1\n')
    (data / 'stat.txt').write_text('6 2\n')
    for split, times in [('train', range(3)), ('valid', [3]), ('test', [4])]:
        rows = [(0, 0, 1, t) for t in times]
        rows += [(0, 0, 2, t) for t in times]
        rows += [(2, 1, 3, t) for t in times]
        rows += [(4, 1, 5, t) for t in times]
        (data / (split + '.txt')).write_text(''.join('{} {} {} {}\n'.format(*r) for r in rows))
    (data / 'e-w-graph-llm.txt').write_text('0 0 6\n1 0 6\n2 0 7\n3 0 7\n')
    return data.parent


def smoke_args(data_root, run_dir, *extra):
    return parse_opts(['-d', 'TINY', '--data-root', str(data_root), '--gpu', '-1',
                       '--run-dir', str(run_dir), '--n-hidden', '16', '--crm-hidden', '8',
                       '--state-rank-k', '2', '--state-rank-v', '4', '--train-history-len', '2',
                       '--dropout', '0', '--report-subgroups', '--n-epochs', '2',
                       '--add-static-graph', '--lr-scheduler', 'plateau',
                       '--lr-scheduler-patience', '0'] + list(extra))


def quiet_run(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return run(args)


class CommandTests(unittest.TestCase):
    def test_history_defaults_follow_dataset_evidence(self):
        for dataset, expected in [('ICEWS14', 7), ('ICEWS05-15', 14), ('ICEWS18', 7), ('YAGO', 7)]:
            with self.subTest(dataset=dataset):
                self.assertEqual(parse_opts(['-d', dataset]).train_history_len, expected)
        self.assertEqual(parse_opts(['-d', 'ICEWS05-15', '--train-history-len', '3']).train_history_len, 3)

    def test_unverified_windows_require_explicit_value_for_fresh_training(self):
        for dataset in ('GDELT', 'WIKI', 'CUSTOM'):
            with self.subTest(dataset=dataset):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_opts(['-d', dataset])
                self.assertEqual(parse_opts(['-d', dataset, '--train-history-len', '10']).train_history_len, 10)

    def test_unverified_window_can_be_inherited_from_checkpoint(self):
        for mode in ('--test', '--resume'):
            args = parse_opts(['-d', 'GDELT', mode, '--load-model', 'models/example/best.pt'])
            self.assertIsNone(args.train_history_len)

    def test_baseline_style_command_and_local_defaults(self):
        args = parse_opts(['-d', 'ICEWS18', '--train-history-len', '7', '--lr', '0.0005',
                           '--n-layers', '2', '--gpu', '0', '--n-hidden', '256',
                           '--decoder', 'convtranse', '--encoder', 'uvrgcn', '--layer-norm',
                           '--pre-weight', '0.7', '--pre-type', 'TF', '--add-static-graph',
                           '--add_query', '--test-every', '0', '--run-name', 'icews18_s42'])
        self.assertEqual(Path(args.data_root), Path('data'))
        self.assertEqual(args.train_order, 'chronological')
        self.assertEqual(args.static_profile, 'fixed_weight')
        self.assertEqual(args.early_stopping_patience, 10)
        self.assertEqual(args.tie_policy, 'torch_original')
        self.assertEqual(args.seed, 42)
        self.assertTrue(args.add_query)
        self.assertFalse(args.enable_c2)
        self.assertFalse(args.use_long_slots)
        self.assertEqual(args.state_gate, 'input')

    def test_test_checkpoint_alias(self):
        args = parse_opts(['--test', '--load-model', './models/example/best.pt'])
        self.assertEqual(args.checkpoint, './models/example/best.pt')

    def test_unsupported_modes_explain_the_limitation(self):
        for option in ['--relation-prediction', '--add-his-graph']:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_opts([option])


class NumericTests(unittest.TestCase):
    def test_extreme_logits_produce_finite_loss_and_gradients(self):
        scores = torch.tensor([[1000., -1000., 0.], [-1000., 1000., 0.]], requires_grad=True)
        targets = torch.tensor([1, 0])
        loss = BaselineModel.task_loss(scores, targets)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(loss.item(), 2000.)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all().item())


class DataTests(unittest.TestCase):
    def test_all_six_local_train_valid_datasets(self):
        expected = {'ICEWS14': (7128, 230), 'ICEWS18': (23033, 256),
                    'ICEWS05-15': (10488, 251), 'GDELT': (7691, 240),
                    'YAGO': (10623, 10), 'WIKI': (12554, 24)}
        for name, axes in expected.items():
            with self.subTest(dataset=name):
                dataset = load_dataset(ROOT / 'data', name)
                self.assertEqual((dataset.meta.N, dataset.meta.R), axes)
                self.assertIsNone(dataset.test)
                self.assertTrue(dataset.train and dataset.valid)
                self.assertEqual(dataset.meta.static.available, name != 'WIKI')
                self.assertGreater(resolve_time_unit(name, None), 0)


class LifecycleTests(unittest.TestCase):
    def test_train_resume_and_minimal_test_restore_static_identity(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / 'tests')) as directory:
            root = Path(directory)
            data = fixture(root)
            full_dir, split_dir, test_dir = root / 'full', root / 'split', root / 'test'
            full = quiet_run(smoke_args(data, full_dir))
            quiet_run(smoke_args(data, split_dir, '--n-epochs', '1'))
            resumed = quiet_run(parse_opts(['-d', 'TINY', '--data-root', str(data), '--gpu', '-1',
                                           '--resume', '--load-model', str(split_dir / 'last.pt'),
                                           '--run-dir', str(split_dir), '--n-epochs', '2']))
            a = torch.load(full_dir / 'last.pt', map_location='cpu')
            b = torch.load(split_dir / 'last.pt', map_location='cpu')
            self.assertEqual(full['progress']['global_step'], resumed['progress']['global_step'])
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertEqual(a['scheduler'], b['scheduler'])
            for identifier, state in a['optimizer']['state'].items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, b['optimizer']['state'][identifier][key], rtol=0, atol=0)
                    else:
                        self.assertEqual(value, b['optimizer']['state'][identifier][key])
            self.assertTrue(torch.equal(a['rng']['torch'], b['rng']['torch']))
            self.assertFalse(full['test_labels_loaded'])
            self.assertFalse((full_dir / 'test.json').exists())
            epochs = [json.loads(line) for line in (full_dir / 'epoch_metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(epochs), 2)
            for epoch in epochs:
                self.assertAlmostEqual(epoch['last']['total'], epoch['last']['main'], places=6)
                self.assertGreater(epoch['average_loss'], 0)
            tested = quiet_run(parse_opts(['-d', 'TINY', '--data-root', str(data), '--gpu', '-1',
                                          '--test', '--load-model', str(full_dir / 'best.pt'),
                                          '--run-dir', str(test_dir), '--result-dir', str(root / 'result')]))
            self.assertEqual(tested['test']['facts'], 8)
            self.assertFalse(tested['test']['partial'])
            self.assertIn('raw', tested['test'])
            self.assertIn('forward', tested['test'])
            self.assertIn('inverse', tested['test'])
            self.assertTrue((root / 'result/TINY.csv').is_file())
            self.assertAlmostEqual(tested['test']['mrr'],
                                   (tested['test']['forward']['mrr'] + tested['test']['inverse']['mrr']) / 2,
                                   places=6)
            config = json.loads((test_dir / 'config.json').read_text())
            self.assertTrue(config['args']['add_static_graph'])
            with self.assertRaises(FileExistsError):
                quiet_run(smoke_args(data, full_dir))

    def test_changed_explicit_resume_mechanism_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / 'tests')) as directory:
            root = Path(directory)
            data = fixture(root)
            quiet_run(smoke_args(data, root / 'run', '--n-epochs', '1'))
            with self.assertRaisesRegex(ValueError, 'state_mode'):
                quiet_run(parse_opts(['-d', 'TINY', '--data-root', str(data), '--gpu', '-1',
                                     '--resume', '--checkpoint', str(root / 'run/last.pt'),
                                     '--state-mode', 'off']))

    def test_real_main_cli_creates_local_models_and_result(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / 'tests')) as directory:
            root = Path(directory)
            fixture(root)
            command = [sys.executable, '-B', str(ROOT / 'main.py'), '-d', 'TINY', '--gpu', '-1']
            trained = subprocess.run(command + ['--run-name', 'tiny_cli', '--n-epochs', '1',
                '--n-hidden', '16', '--crm-hidden', '8', '--state-rank-k', '2', '--state-rank-v', '4',
                '--train-history-len', '2', '--add-static-graph', '--add_query'],
                cwd=str(root), capture_output=True, text=True, encoding='utf-8', errors='replace')
            self.assertEqual(trained.returncode, 0, trained.stderr)
            self.assertTrue((root / 'models/tiny_cli/best.pt').is_file())
            tested = subprocess.run(command + ['--test', '--load-model', 'models/tiny_cli/best.pt'],
                cwd=str(root), capture_output=True, text=True, encoding='utf-8', errors='replace')
            self.assertEqual(tested.returncode, 0, tested.stderr)
            self.assertTrue((root / 'result/TINY.csv').is_file())
            self.assertEqual(json.loads(tested.stdout.splitlines()[-1])['model'], 'CRM-DER')


if __name__ == '__main__':
    unittest.main()
