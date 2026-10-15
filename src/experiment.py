"""Local run layout and CSV output for the standalone CRM-DER project."""
import csv
from datetime import datetime
from pathlib import Path


def restore_arguments(args, checkpoint):
    """Use saved training semantics; permit operational options on test/resume.

    Explicit resume changes are validated by the trainer before this merge.
    In particular static mode must be restored BEFORE hashing data assets.
    """
    import argparse
    current = vars(args).copy()
    saved = current.copy()
    saved.update(checkpoint['config']['args'])
    if saved['dataset'] != args.dataset:
        raise ValueError('Checkpoint dataset differs from requested dataset')
    for key in ('test', 'resume', 'checkpoint', 'gpu', 'data_root', 'cpu_threads',
                'max_steps', 'max_valid_frames', 'smoke_target_limit', 'memory_budget_gib',
                'common_state', 'export_common_state', 'result_dir', '_explicit'):
        saved[key] = current[key]
    for key in ('n_epochs', 'early_stopping_patience', 'report_subgroups', 'run_name'):
        if key in args._explicit:
            saved[key] = current[key]
    # Never inherit a saved output directory when evaluating a checkpoint.
    saved['run_dir'] = current['run_dir']
    if args.resume and saved['run_dir'] is None:
        saved['run_dir'] = str(Path(args.checkpoint).absolute().parent)
    return argparse.Namespace(**saved)


def configure_output(args):
    if args.run_name is None:
        args.run_name = 'CRM-DER-{}-h{}-len{}-seed{}-{}'.format(
            args.dataset, args.n_hidden, args.train_history_len, args.seed,
            datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if args.run_dir is None:
        if args.test:
            args.run_dir = str(Path(args.result_dir) / (args.run_name + '-test-' +
                                                       datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        else:
            args.run_dir = str(Path('models') / args.run_name)
    directory = Path(args.run_dir)
    if not args.resume and not args.test and (directory / 'last.pt').exists():
        raise FileExistsError('Training output already exists; use --resume or a new --run-name: ' + str(directory))
    if args.test and (directory / 'last.pt').exists():
        raise FileExistsError('Test output must use a separate directory from training: ' + str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_test_csv(args, result, progress):
    metrics = result['test']
    row = dict(model='CRM-DER', dataset=args.dataset, seed=args.seed, run_name=args.run_name,
               checkpoint_path=str(Path(args.checkpoint).absolute()),
               checkpoint_epoch=progress['epoch'], best_valid_mrr=progress['best_mrr'],
               datetime=datetime.now().isoformat(timespec='seconds'),
               encoder=args.encoder, decoder=args.decoder, pre_type=args.pre_type,
               use_static=args.add_static_graph, pre_weight=args.pre_weight,
               train_len=args.train_history_len, n_hidden=args.n_hidden, lr=args.lr,
               static_loss_weight=args.static_weight, static_profile=args.static_profile,
               static_warmup_epochs=args.static_warmup_epochs, train_order=args.train_order,
               tie_policy=args.tie_policy, state_gate=args.state_gate, state_mode=args.state_mode,
               partial=metrics['partial'], facts=metrics['facts'], frames=metrics['frames'])
    for prefix, group in [('filter_all', metrics), ('all', metrics['raw']),
                          ('filter', metrics['forward']), ('filter_inv', metrics['inverse'])]:
        for name, key in [('MRR', 'mrr'), ('H@1', 'hits1'), ('H@3', 'hits3'), ('H@10', 'hits10')]:
            row[prefix + '_' + name] = group[key]
    for group in ('repeat', 'A', 'B'):
        row[group + '_MRR'] = metrics.get('subgroups', {}).get(group, {}).get('mrr')
    directory = Path(args.result_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (args.dataset + '.csv')
    exists = path.is_file() and path.stat().st_size > 0
    if exists:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            if next(csv.reader(handle)) != list(row):
                raise ValueError('CSV header differs; choose a separate --result-dir: ' + str(path))
    with path.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return str(path)
