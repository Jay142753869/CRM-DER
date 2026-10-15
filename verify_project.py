"""Validate local assets; --smoke adds a small CPU forward/backward per dataset.

No real test labels are parsed, no optimizer is created, no weights are saved.
Smoke uses D32/H3 and is an integration check, not a performance experiment.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parent
sys.path.insert(0, str(ROOT))


def verify(smoke=False):
    import torch
    from opt import parse_opts
    from src.crm.factory import build_model, resolve_time_unit
    from src.data import load_dataset
    from src.evaluation import known_direction
    from src.runtime import seed_all
    from src.trainer import Runtime

    torch.set_num_threads(1)
    manifest = json.loads((ROOT / 'docs/DATA_MANIFEST.json').read_text(encoding='utf-8'))
    results = {}
    for name, files in manifest.items():
        print('Checking {} ...'.format(name), flush=True)
        for filename, record in files.items():
            path = ROOT / 'data' / name / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Asset differs from packaged source: ' + str(path))
        dataset = load_dataset(ROOT / 'data', name)
        record = dict(entities=dataset.meta.N, relations=dataset.meta.R,
                      train_frames=len(dataset.train), valid_frames=len(dataset.valid),
                      static_available=dataset.meta.static.available,
                      time_unit=resolve_time_unit(name, None), test_labels_loaded=dataset.test is not None)
        if smoke:
            options = ['-d', name, '--data-root', str(ROOT / 'data'), '--gpu', '-1',
                       '--n-hidden', '32', '--crm-hidden', '16', '--train-history-len', '3',
                       '--state-rank-k', '2', '--state-rank-v', '8']
            if dataset.meta.static.available:
                options += ['--add-static-graph']
            args = parse_opts(options)
            seed_all(args.seed)
            model = build_model(dataset.meta, args)
            runtime = Runtime(dataset, args, torch.device('cpu'))
            # A real, complete snapshot and candidate axis, with >=2 past frames.
            frame = min(dataset.train[2:20], key=lambda f: len(f.facts))
            graphs, times = runtime.history(frame, True)
            requests, labels = known_direction(frame.facts, dataset.meta.R, 'entity')
            output = model(torch.tensor(requests), graphs, times, frame.timestamp, runtime.static_graph)
            loss = model.task_loss(output.raw_scores, torch.tensor(labels))
            if output.raw_scores.shape != (len(labels), dataset.meta.N) or not torch.isfinite(loss):
                raise AssertionError('Invalid full-candidate forward for ' + name)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise AssertionError('Non-finite or missing gradients for ' + name)
            record['cpu_smoke'] = dict(hidden=32, history=3, timestamp=frame.timestamp,
                                       rows=len(labels), candidates=dataset.meta.N,
                                       loss=float(loss.detach()), recent_slots=output.stats['recent_slots'],
                                       state_slots=output.stats['state_slots'], optimizer_created=False)
            del loss, output, gradients, model, runtime, graphs
            gc.collect()
        results[name] = record
        print(json.dumps(record), flush=True)
    destination = ROOT / 'docs' / ('DATA_SMOKE_RESULTS.json' if smoke else 'DATA_CHECK_RESULTS.json')
    destination.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    verify(parser.parse_args().smoke)
