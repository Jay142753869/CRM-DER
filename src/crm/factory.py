# -*- coding: utf-8 -*-
"""CRM-DER 模型工厂：唯一构造入口，负责语义开关与 provenance 记录。"""
from pathlib import Path

from ..backbone.initialization import isolated_module
from .model import CRMModel

# 数据集时间单位表（原始 ticks / unit）。仅作显式默认，可在 CLI 覆盖并写入 checkpoint。
# 声明值是每个名义快照单位的原生整数 ticks；不据此推断小时、分钟或天。
TIME_UNIT_TABLE = {'ICEWS14': 1.0, 'ICEWS14s': 1.0, 'ICEWS18': 24.0, 'GDELT': 15.0,
                   'ICEWS05-15': 1.0, 'YAGO': 1.0, 'WIKI': 1.0}


def resolve_time_unit(dataset, explicit):
    if explicit is not None:
        if explicit <= 0:
            raise ValueError('time-unit must be positive')
        return float(explicit)
    return float(TIME_UNIT_TABLE.get(dataset, 1.0))


def build_model(meta, args):
    """构造正式模型。所有语义开关在此集中校验，避免静默降级。"""
    if args.task != 'entity':
        raise ValueError('CRM-DER v2 首版只支持 entity 任务；relation prediction 未实现')
    if args.add_static_graph and not meta.static.available:
        raise ValueError('Static graph unavailable; choose explicit no-static mode')
    if args.pre_type != 'TF':
        raise ValueError('CRM-DER 只验证过 TF；GRU 模式未验证')
    if args.use_long_slots:
        raise ValueError(
            'long_slots 首版关闭：raw old E×R 与近期 RGCN 消息不是同语义，'
            '需要独立 source adapter；本版本拒绝静默启用（scope_revision_pending_review）')
    if args.crm_hidden <= 0 or args.state_rank_k <= 0 or args.state_rank_v <= 0:
        raise ValueError('CRM 维度必须为正')
    args.time_unit = resolve_time_unit(args.dataset, args.time_unit)
    return CRMModel(meta, args)


def parameter_report(model):
    """参数量分账：主干 / 状态 / 读出 / C2，逐项单列。"""
    backbone = sum(p.numel() for p in model.backbone.parameters())
    state = sum(p.numel() for p in model.state.parameters()) if model.state is not None else 0
    readout = sum(p.numel() for p in model.readout.parameters())
    c2 = sum(p.numel() for p in model.c2.parameters()) if model.c2 is not None else 0
    total = sum(p.numel() for p in model.parameters())
    return dict(backbone=backbone, state=state, readout=readout, c2=c2,
                crm_module=state + readout + c2, total=total,
                crm_share_of_total=round((state + readout + c2) / max(total, 1), 6),
                state_mode=model.args.state_mode,
                gate_mode=None if model.state is None else model.state.gate_mode,
                state_gate_params=(sum(p.numel() for p in model.state.gate_parameters())
                                   if model.state is not None else 0))
