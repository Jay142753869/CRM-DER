"""CRM-DER CLI。导入本模块不解析进程参数。"""
import argparse
import math
import sys

# QDRL Fig. 2 (PDF p14 / printed p420); see docs/HISTORY_WINDOWS.md.
# These are starting settings, not demonstrated CRM-DER optima.
DEFAULT_HISTORY_LENGTHS = {'ICEWS14': 7, 'ICEWS14s': 7, 'ICEWS05-15': 14,
                           'ICEWS18': 7, 'YAGO': 7}


def parser():
    p = argparse.ArgumentParser(description='CRM-DER v2: shared causal event bank + dual-end readout')
    p.add_argument('-d', '--dataset', default='ICEWS14')
    p.add_argument('--data-root', default='./data')
    p.add_argument('--task', choices=['entity'], default='entity')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--n-hidden', type=int, default=256)
    p.add_argument('--train-history-len', type=int, default=None,
                   help='snapshot count: ICEWS14=7, ICEWS05-15=14, ICEWS18=7, YAGO=7; '
                        'other datasets require an explicit value for fresh training')
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--n-epochs', type=int, default=50)
    p.add_argument('--early-stopping-patience', type=int, default=10)
    p.add_argument('--evaluate-every', type=int, default=1)
    p.add_argument('--test-every', type=int, choices=[0], default=0,
                   help='test is explicit only: use --test --load-model after training')
    p.add_argument('--decoder', choices=['convtranse'], default='convtranse')
    p.add_argument('--encoder', choices=['uvrgcn'], default='uvrgcn')
    p.add_argument('--pre-type', choices=['TF'], default='TF')
    p.add_argument('--q-type', choices=['conv'], default='conv', help='R02 uses ConvTransE conv queries')
    p.add_argument('--self-loop', action='store_true', default=True, help='R02 self loops are enabled')
    p.add_argument('--pre-weight', type=float, default=.7)
    p.add_argument('--layer-norm', action='store_true', default=True)
    p.add_argument('--no-layer-norm', dest='layer_norm', action='store_false')
    p.add_argument('--add-query', '--add_query', dest='add_query', action='store_true', default=True)
    p.add_argument('--no-query', dest='add_query', action='store_false')
    p.add_argument('--add-static-graph', action='store_true')
    p.add_argument('--static-profile', choices=['baseline_threshold', 'fixed_weight', 'off'],
                   default='fixed_weight')
    p.add_argument('--static-weight', '--static-loss-weight', dest='static_weight', type=float, default=.1)
    p.add_argument('--static-warmup-epochs', type=int, default=0)
    p.add_argument('--static-score-chunk', type=int, default=256)
    p.add_argument('--dropout', type=float, default=.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run-name', help='name under ./models; generated when omitted')
    p.add_argument('--run-dir', help='override ./models/<run-name>')
    p.add_argument('--result-dir', default='./result')
    p.add_argument('--checkpoint', '--load-model', dest='checkpoint')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--test', action='store_true')
    p.add_argument('--common-state', help='配对初始化用的未训练主干 state_dict')
    p.add_argument('--export-common-state', help='导出初始化后的主干 state_dict')
    p.add_argument('--cpu-threads', type=int, default=1)
    p.add_argument('--train-order', choices=['shuffled', 'chronological'], default='chronological')
    p.add_argument('--tie-policy', choices=['stable_id', 'torch_original'], default='torch_original')
    p.add_argument('--report-subgroups', action='store_true')
    p.add_argument('--grad-norm', type=float, default=1.)
    p.add_argument('--lr-scheduler', choices=['none', 'plateau'], default='none')
    p.add_argument('--lr-scheduler-patience', type=int, default=3)
    p.add_argument('--lr-scheduler-factor', type=float, default=.5)
    p.add_argument('--max-steps', type=int, help='显式 partial train smoke 步数上限')
    p.add_argument('--max-valid-frames', type=int, help='显式 partial validation 帧数上限')
    p.add_argument('--smoke-target-limit', type=int, help='partial smoke 的目标上限')
    p.add_argument('--memory-budget-gib', type=float, default=2.5)
    # ---- CRM-DER 专属 ----
    p.add_argument('--time-unit', type=float, default=None,
                   help='原始 ticks / 声明时间单位；缺省查数据集表，绝不从 test 推断')
    p.add_argument('--state-gate', choices=['input', 'global'], default='input',
                   help='input=输入依赖门控（主规格）；global=同信息简化控制')
    p.add_argument('--state-mode', choices=['on', 'off'], default='on',
                   help='off = A-nostate 消融臂：不构造因果状态模块，也不注入状态槽')
    p.add_argument('--state-rank-k', type=int, default=8)
    p.add_argument('--state-rank-v', type=int, default=32)
    p.add_argument('--crm-hidden', type=int, default=64)
    p.add_argument('--crm-query-chunk', type=int, default=32)
    p.add_argument('--no-crm-checkpoint-chunks', dest='crm_checkpoint_chunks', action='store_false', default=True,
                   help='disable exact candidate-block activation recomputation (memory control)')
    p.add_argument('--gamma-subj-init', type=float, default=0.01)
    p.add_argument('--enable-c2', action='store_true',
                   help='显式恢复独立 C2（H_raw 双线性时间选择）对照臂；默认关闭')
    p.add_argument('--c2-hidden', type=int, default=64)
    p.add_argument('--c2-time-bias', action='store_true')
    p.add_argument('--use-long-slots', action='store_true',
                   help='长期槽首版关闭；本版本拒绝启用（需独立 source adapter）')
    p.add_argument('--relation-prediction', action='store_true', help='unsupported by R02 entity model')
    p.add_argument('--add-his-graph', action='store_true', help='unsupported: CRM uses its own event bank')
    return p


def parse_opts(argv=None):
    p = parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(tokens)
    args._explicit = sorted({p._option_string_actions[token.split('=', 1)[0]].dest
                             for token in tokens if token.split('=', 1)[0] in p._option_string_actions})
    if args.train_history_len is None and not (args.test or args.resume):
        args.train_history_len = DEFAULT_HISTORY_LENGTHS.get(args.dataset)
        if args.train_history_len is None:
            p.error('No verified history window for {}; supply --train-history-len <snapshot count>. '
                    'See docs/HISTORY_WINDOWS.md; test/resume inherit the saved window.'.format(args.dataset))
    if args.relation_prediction or args.add_his_graph:
        p.error('CRM-DER (R02) supports entity prediction and its own event bank; '
                '--relation-prediction/--add-his-graph are not implemented')
    if args.smoke_target_limit:
        p.error('--smoke-target-limit is not implemented; use --max-steps/--max-valid-frames')
    if args.run_name and (args.run_name in ('.', '..') or any(c in args.run_name for c in '/\\:')):
        p.error('run-name must be one directory name')
    if args.static_warmup_epochs < 0 or args.lr_scheduler_patience < 0:
        p.error('warmup and scheduler patience must be nonnegative')
    if not math.isfinite(args.static_weight) or args.static_weight < 0:
        p.error('static-loss-weight must be finite and nonnegative')
    if not 0 < args.lr_scheduler_factor < 1:
        p.error('lr-scheduler-factor must be between zero and one')
    for key in ('n_hidden', 'train_history_len', 'n_layers', 'n_epochs', 'evaluate_every',
                'state_rank_k', 'state_rank_v', 'crm_hidden', 'crm_query_chunk', 'c2_hidden',
                'static_score_chunk'):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            p.error(key + ' must be positive')
    if args.early_stopping_patience < 0:
        p.error('early-stopping-patience must be nonnegative')
    if args.early_stopping_patience and (args.evaluate_every != 1 or args.max_steps or args.max_valid_frames or args.smoke_target_limit):
        p.error('early stopping requires every complete epoch and full validation; use --early-stopping-patience 0 for partial smoke')
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error('lr must be positive and finite')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error('weight-decay must be finite and nonnegative')
    if args.time_unit is not None and (not math.isfinite(args.time_unit) or args.time_unit <= 0):
        p.error('time-unit must be positive and finite')
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib <= 0:
        p.error('memory-budget-gib must be positive and finite')
    if not math.isfinite(args.gamma_subj_init):
        p.error('gamma-subj-init must be finite')
    for key in ('max_steps', 'max_valid_frames', 'smoke_target_limit'):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            p.error(key + ' must be positive when supplied')
    if (args.test or args.resume) and not args.checkpoint:
        p.error('--test/--resume requires --checkpoint')
    if args.test and args.resume:
        p.error('--test and --resume are mutually exclusive')
    if (args.test or args.resume) and (args.common_state or args.export_common_state):
        p.error('common-state initialization/export is only for fresh training')
    return args
