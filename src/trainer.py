"""CRM-DER trainer: fresh / resume / explicit test, strict checkpoint semantics."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .backbone.graph import build_graph
from .crm.factory import build_model, parameter_report, resolve_time_unit
from .data import load_dataset, load_static_graph, PastIndex, make_frames
from .evaluation import filtered_ranks, known_direction, preceding_frames, ranking_metrics
from .runtime import code_hash, load_checkpoint, save_checkpoint, seed_all
from .experiment import configure_output, restore_arguments, write_test_csv

RESUME_KEYS = ('dataset', 'task', 'n_hidden', 'train_history_len', 'n_layers', 'seed',
               'pre_type', 'pre_weight', 'layer_norm', 'add_query', 'dropout',
               'add_static_graph', 'static_profile', 'static_weight',
               'state_gate', 'state_mode', 'state_rank_k', 'state_rank_v', 'crm_hidden',
               'crm_query_chunk', 'crm_checkpoint_chunks', 'gamma_subj_init', 'enable_c2', 'c2_hidden',
               'c2_time_bias', 'use_long_slots', 'time_unit', 'lr', 'weight_decay',
               'grad_norm', 'tie_policy', 'train_order', 'static_warmup_epochs',
               'lr_scheduler', 'lr_scheduler_patience', 'lr_scheduler_factor')


def data_identity(args, meta):
    names = ['train.txt', 'valid.txt', 'entity2id.txt', 'relation2id.txt', 'stat.txt']
    if args.add_static_graph:
        names.append('e-w-graph-llm.txt')
    hashes = {}
    for name in names:
        digest = hashlib.sha256()
        with (Path(args.data_root) / meta.name / name).open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return dict(dataset=meta.name, N=meta.N, R=meta.R, files=hashes)


def checkpoint_config(args, model):
    return dict(
        arch_version='crm-der-v2',
        model_name='CRM-DER',
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        constructors=model.constructors(),
        code_hash=code_hash(),
        protocol=dict(
            directions='forward_then_inverse_separate_optimizer_updates',
            history='strictly_past_real_timestamps_only',
            validation='rolling_observed_strict_real_prefix',
            test='explicit_checkpoint_only; train/valid_never_parse_test',
            objective='stable_log_softmax_nll' + (' + static:' + args.static_profile if args.add_static_graph else ''),
            training_reference='QDRL-main-new: chronological, fixed static weight, patience10 defaults',
            partial=bool(args.max_steps or args.max_valid_frames or args.smoke_target_limit),
            full_candidate_axis=True,
            neural_state='recomputed_inside_each_forward; no cross-sample cache'))


def validate_resume_identity(args, config):
    saved = config['args']
    if config.get('code_hash') != code_hash():
        raise ValueError('Checkpoint code hash differs; CRM mechanism source changed')
    changed = [key for key in RESUME_KEYS
               if key in saved and key in args._explicit and getattr(args, key, saved[key]) != saved[key]]
    if changed:
        raise ValueError('Checkpoint resume identity mismatch: ' + ', '.join(changed))
    constructors = config.get('constructors', {})
    if constructors.get('sharing', {}).get('keys_values') != 'single_prepare_shared_by_both_ends':
        raise ValueError('Checkpoint sharing semantics differ from this implementation')


class Runtime:
    """Frames, graphs and optional static background; no neural activation cache."""

    def __init__(self, dataset, args, device, allow_test=False):
        if dataset.test is not None and not allow_test:
            raise ValueError('Test frames require explicit final evaluation')
        self.dataset, self.args, self.device = dataset, args, device
        self.train_frames = tuple(sorted(dataset.train, key=lambda f: f.timestamp))
        observed = dataset.train + dataset.valid + (dataset.test or ())
        if len({f.timestamp for f in observed}) < len(observed):
            self.eval_frames = make_frames(np.concatenate([f.facts for f in observed]), 'observed')
        else:
            self.eval_frames = tuple(sorted(observed, key=lambda f: f.timestamp))
        self.graph_cache = {}
        self.static_graph = None
        self.static_triples = None
        if args.add_static_graph:
            static = load_static_graph(Path(args.data_root) / dataset.meta.name, dataset.meta.N)
            triples = torch.tensor(np.array(static.triples), dtype=torch.long, device=device)
            self.static_graph = build_graph(static.meta.num_nodes, static.meta.relation_axis,
                                            triples).to(device)
            self.static_triples = triples
        self.index = PastIndex(self.eval_frames, dataset.meta.N, dataset.meta.R,
                               allow_test=allow_test)

    def history(self, frame, training):
        chosen = preceding_frames(self.train_frames if training else self.eval_frames,
                                  frame.timestamp, self.args.train_history_len)
        graphs = []
        needed = {(f.split, f.timestamp) for f in chosen}
        self.graph_cache = {k: v for k, v in self.graph_cache.items() if k in needed}
        for f in chosen:
            key = (f.split, f.timestamp)
            if key not in self.graph_cache:
                triples = torch.tensor(np.array(f.facts[:, :3]), dtype=torch.long)
                self.graph_cache[key] = build_graph(self.dataset.meta.N, self.dataset.meta.R,
                                                    triples).to(self.device)
            graphs.append(self.graph_cache[key])
        return graphs, [f.timestamp for f in chosen]


def _to_tensor(values, device):
    return torch.tensor(values, dtype=torch.long, device=device)


def static_loss_weight(args, epoch):
    ratio = min(1., float(epoch + 1) / args.static_warmup_epochs) if args.static_warmup_epochs else 1.
    return args.static_weight * ratio


def train_step(model, optimizer, runtime, frame, direction, args, device, epoch=0):
    graphs, times = runtime.history(frame, True)
    if not graphs:
        return None
    requests, targets = known_direction(frame.facts, runtime.dataset.meta.R, 'entity', direction)
    requests = _to_tensor(requests, device)
    targets = _to_tensor(targets, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(requests, graphs, times, frame.timestamp, runtime.static_graph)
    main = model.task_loss(output.raw_scores, targets)
    loss = main
    static_value = None
    if args.add_static_graph and args.static_profile != 'off' and not direction:
        static_value = model.static_loss(output.history, runtime.static_triples)
        if args.static_profile == 'baseline_threshold':
            loss = model.backbone.original_total_loss(main, static_value)
        else:
            loss = main + static_loss_weight(args, epoch) * static_value
    if not torch.isfinite(loss).all():
        raise FloatingPointError('Non-finite loss at timestamp {}, inverse={}'.format(frame.timestamp, direction))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm, error_if_nonfinite=True)
    optimizer.step()
    record = dict(main=float(main.detach()), total=float(loss.detach()))
    if static_value is not None:
        record['static'] = float(static_value.detach())
    record.update({k: v for k, v in output.stats.items() if isinstance(v, (int, float, str))})
    return record


@torch.no_grad()
def evaluate(model, runtime, frames, args, limit=None):
    model.eval()
    ranks = []
    raw_ranks = []
    direction_ranks = [[], []]
    subgroups = {key: [] for key in ('repeat', 'A', 'B')}
    selected = frames[:limit] if limit else frames
    frames_done = 0
    for frame in selected:
        graphs, times = runtime.history(frame, False)
        if not graphs:
            continue
        view = runtime.index.before(frame.timestamp)
        for direction in (False, True):
            requests, targets = known_direction(frame.facts, runtime.dataset.meta.R, 'entity', direction)
            requests = _to_tensor(requests, device=runtime.device)
            targets = _to_tensor(targets, device=runtime.device)
            output = model(requests, graphs, times, frame.timestamp, runtime.static_graph)
            current = filtered_ranks(output.raw_scores, output.requests, targets,
                                     tie_policy=args.tie_policy).cpu()
            ranks.append(current)
            direction_ranks[int(direction)].append(current)
            scores = output.raw_scores
            if args.tie_policy == 'torch_original':
                raw_rank = 1 + (torch.argsort(scores, dim=1, descending=True) == targets[:, None]).long().argmax(1)
            else:
                true = scores.gather(1, targets[:, None])
                ids = torch.arange(scores.shape[1], device=scores.device)
                raw_rank = 1 + (scores > true).sum(1) + ((scores == true) & (ids < targets[:, None])).sum(1)
            raw_ranks.append(raw_rank.cpu())
            if args.report_subgroups:
                groups = {'repeat': [], 'A': [], 'B': []}
                for row, ((s, r), target) in enumerate(zip(output.requests.cpu().tolist(),
                                                           targets.cpu().tolist())):
                    kind = ('repeat' if target in view.objects(s, r)
                            else 'A' if target in view.neighbors(s) else 'B')
                    groups[kind].append(row)
                for key, rows in groups.items():
                    subgroups[key].append(current[rows])
        frames_done += 1
    result = ranking_metrics(torch.cat(ranks) if ranks else torch.empty(0))
    result.update(frames=frames_done, partial=bool(limit or args.max_valid_frames or args.max_steps),
                  tie_policy=args.tie_policy)
    result['raw'] = ranking_metrics(torch.cat(raw_ranks) if raw_ranks else torch.empty(0))
    for key, values in zip(('forward', 'inverse'), direction_ranks):
        result[key] = ranking_metrics(torch.cat(values) if values else torch.empty(0))
    if args.report_subgroups:
        result['subgroups'] = {key: ranking_metrics(torch.cat(v) if v else torch.empty(0))
                               for key, v in subgroups.items()}
    return result


def run(args):
    args.time_unit = resolve_time_unit(args.dataset, args.time_unit)
    checkpoint = None
    if args.checkpoint and (args.resume or args.test):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if not isinstance(checkpoint, dict) or checkpoint.get('arch_version') != 'crm-der-v2':
            raise ValueError('Expected a CRM-DER/R02 checkpoint, not a baseline model')
        if args.resume:
            validate_resume_identity(args, checkpoint['config'])
        args = restore_arguments(args, checkpoint)
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; pass --gpu -1 for CPU')
    device = torch.device('cuda:{}'.format(args.gpu) if args.gpu >= 0 else 'cpu')
    directory = configure_output(args)
    dataset = load_dataset(args.data_root, args.dataset, include_test=args.test)
    identity = data_identity(args, dataset.meta)
    if len(dataset.train) < 2:
        raise ValueError('Training data needs at least two distinct historical timestamps')
    model = build_model(dataset.meta, args).to(device)
    model.backbone.decoder_ob.pre_weight.data.fill_(args.pre_weight)
    optimizer = None if args.test else torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_scheduler_factor, patience=args.lr_scheduler_patience)
        if optimizer is not None and args.lr_scheduler == 'plateau' else None)
    if args.export_common_state:
        path = Path(args.export_common_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.backbone.state_dict(), path)
    progress = dict(epoch=0, global_step=0, order=None, cursor=0, direction=0,
                    best_mrr=-1.0, best_epoch=None)
    if checkpoint is not None:
        payload = load_checkpoint(args.checkpoint, model, optimizer if args.resume else None,
                                  restore_rng=args.resume, expected_data=identity)
        progress = payload['progress']
        if args.resume and scheduler is not None:
            if payload.get('scheduler') is None:
                raise ValueError('Checkpoint is missing the requested scheduler state')
            scheduler.load_state_dict(payload['scheduler'])
    runtime = Runtime(dataset, args, device, allow_test=args.test)
    config = checkpoint_config(args, model)
    config['parameters'] = parameter_report(model)
    (directory / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    if args.test:
        metrics = evaluate(model, runtime, dataset.test, args, limit=args.max_valid_frames)
        result = dict(model='CRM-DER', test=metrics, checkpoint=args.checkpoint, progress=progress,
                      parameters=config['parameters'])
        result['csv'] = write_test_csv(args, result, progress)
        (directory / 'test.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
        return result

    updates = 0
    validation = None
    last_record = None
    stop_reason = 'epoch_limit'
    already_stopped = bool(args.early_stopping_patience and progress['best_epoch'] is not None
                           and progress['epoch'] - progress['best_epoch'] >= args.early_stopping_patience)
    if already_stopped:
        stop_reason = 'early_stopping'
    while progress['epoch'] < args.n_epochs and not already_stopped:
        started = time.perf_counter()
        epoch_losses = []
        if progress['order'] is None:
            eligible = [i for i, f in enumerate(dataset.train)
                        if preceding_frames(runtime.train_frames, f.timestamp, args.train_history_len)]
            progress['order'] = (np.random.permutation(eligible).tolist()
                                 if args.train_order == 'shuffled' else eligible)
            progress['cursor'] = 0
            progress['direction'] = 0
        epoch_updates = 0
        print('CRM-DER | epoch {} / {} | {} | snapshots {} | output {}'.format(
            progress['epoch'] + 1, args.n_epochs, args.dataset, len(progress['order']), directory), flush=True)
        while progress['cursor'] < len(progress['order']):
            frame = dataset.train[progress['order'][progress['cursor']]]
            while progress['direction'] < 2:
                last_record = train_step(model, optimizer, runtime, frame,
                                         bool(progress['direction']), args, device, epoch=progress['epoch'])
                if last_record is None:
                    break
                progress['global_step'] += 1
                updates += 1
                epoch_updates += 1
                epoch_losses.append(last_record['total'])
                progress['direction'] += 1
                if args.max_steps and updates >= args.max_steps:
                    break
            if progress['direction'] >= 2:
                progress['direction'] = 0
                progress['cursor'] += 1
                if progress['cursor'] % 50 == 0:
                    print('  snapshots {}/{} | loss {:.4f} | updates {}'.format(
                        progress['cursor'], len(progress['order']), last_record['total'],
                        progress['global_step']), flush=True)
            if args.max_steps and updates >= args.max_steps:
                break
        complete = progress['cursor'] >= len(progress['order'])
        if complete:
            progress['epoch'] += 1
            progress['order'] = None
            progress['cursor'] = 0
            progress['direction'] = 0
        do_validate = (complete and progress['epoch'] % args.evaluate_every == 0) or \
                      bool(args.max_steps and updates >= args.max_steps)
        if do_validate:
            validation = evaluate(model, runtime, dataset.valid, args,
                                  limit=args.max_valid_frames)
            if validation['mrr'] is None or not np.isfinite(validation['mrr']):
                raise FloatingPointError('Validation produced empty or non-finite metrics')
            if scheduler is not None:
                scheduler.step(validation['mrr'])
            if validation['mrr'] >= progress['best_mrr']:
                progress['best_mrr'] = validation['mrr']
                progress['best_epoch'] = progress['epoch']
                save_checkpoint(directory / 'best.pt', model, optimizer, dict(progress),
                                config, identity, scheduler=scheduler)
            average_loss = float(np.mean(epoch_losses)) if epoch_losses else None
            with (directory / 'epoch_metrics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(epoch=progress['epoch'], global_step=progress['global_step'],
                    updates=epoch_updates, last=last_record, average_loss=average_loss,
                    elapsed_seconds=time.perf_counter() - started, lr=optimizer.param_groups[0]['lr'],
                    validation=validation, parameters=config['parameters'])) + '\n')
            print('Epoch {:04d} | loss {} | valid MRR {:.6f} | best {:.6f} (epoch {}) | lr {:.6g}'.format(
                progress['epoch'], average_loss, validation['mrr'], progress['best_mrr'],
                progress['best_epoch'], optimizer.param_groups[0]['lr']), flush=True)
        save_checkpoint(directory / 'last.pt', model, optimizer, dict(progress), config, identity, scheduler=scheduler)
        if args.max_steps and updates >= args.max_steps:
            stop_reason = 'max_steps'
            break
        if args.early_stopping_patience and progress['best_epoch'] is not None:
            if progress['epoch'] - progress['best_epoch'] >= args.early_stopping_patience:
                stop_reason = 'early_stopping'
                break
    result = dict(model='CRM-DER', run_dir=str(directory), progress=progress, validation=validation, last=last_record,
                  updates_this_run=updates, stop_reason=stop_reason,
                  partial=bool(args.max_steps or args.max_valid_frames or args.smoke_target_limit),
                  test_labels_loaded=False, parameters=config['parameters'])
    (directory / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    return result
