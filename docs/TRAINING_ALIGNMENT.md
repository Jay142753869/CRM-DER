# CRM-DER 的模型来源与训练对齐

日期：2026-10-16。用户明确要求独立命名为 CRM-DER，并选择按 QDRL-main-new 的训练协议统一。

## 模型保持 R02

源码基于 `report/review_20261013/rearchitecture_panel/R02/implementation/QDRL` 的正式工程。
复制时的逐文件 SHA256 记录于 SOURCE_MANIFEST.json。CRM 核心、模型、分组、工厂以及 CNN/MA/
ConvTransE 等评分路径保持原实现。唯一的 backbone 改动是将 `log(softmax)` 换为与 baseline 相同的
`log_softmax`，它改变数值稳定性，不改变数学目标或推理 logits。

没有将 baseline 的 main.py 覆盖到 R02。CRM-DER 的 main.py 调用 src.trainer.run，模型工厂始终构造 CRMModel。
初始化为新模型，不复制旧训练权重，不启用 C2 或长期槽。

## 默认协议变化

| 项目 | 历史 R02 正式实验 | 本工程默认 |
|---|---|---|
| seed | 43 | 42（baseline 默认，可指定） |
| 顺序 | shuffled | chronological（与 baseline 相同） |
| 静态损失 | static>3 时 0.9*main+0.1*static，否则 main | main+0.1*static |
| 静态 warmup | 无 | 默认 0，可显式开启 |
| 主/静态 NLL | log(softmax) | log_softmax |
| 早停 | 8 轮无严格改善 | 10 轮下降；相等按 baseline 保存并重置计数 |
| 学习率调度 | 无 | 默认无，可选 plateau，保存并恢复状态 |
| 数据目录 | 指向另一个工程 | 本地 ./data 独立副本 |
| 输出 | 指定 report/formal | ./models/<run-name> 与 ./result |
| test | 显式固定 checkpoint | 显式固定 checkpoint，兼容 --load-model |

与 baseline 一致：Adam、weight decay=1e-5、clip=1、dropout=0.2、D256/L2、TF、ConvTransE、
uvrgcn、正反向分别更新、完整 valid、全部实体候选。查询 token 与 layer norm 在 R02 默认开启，
README 训练命令仍显式携带对应参数。静态监督仅在正向计算，逆向静态项为零，与 baseline 的实际用法一致。

历史窗口更正：不再将 ICEWS14 的 H7 套用全部数据集。按本地论文 Fig. 2，ICEWS05-15 使用 H14；
ICEWS14、ICEWS18、YAGO 使用 H7。正文 §5.6 的数据集名称有重复笔误，后两项以图示为依据。
GDELT 没有对应曲线，WIKI 不在论文五个实验数据集中，因此首次训练要求显式窗口，不能把通用 CLI 默认值当作其论文设置。
详见 HISTORY_WINDOWS.md；checkpoint 的实际窗口始终优先用于 test/resume。

baseline 中 `--test-every` 没有控制末尾 `epoch >= 5` 的测试块；那个块会加载 best 权重到训练中的模型，
后续优化器状态却继续沿用。因此本工程不复制这段训练轨迹干扰。`--test-every` 仅接受 0，
训练不读 test；需要评估时显式 `--test --load-model`。这项修正已在聊天中说明。

数学与代码上的统一不代表随机初始化、模块参数和执行顺序逐位等同于 baseline；R02 本来就是另一个架构。
本工程的新训练应记录为“CRM-DER，baseline 训练协议”，不能冒称复现历史 R02 的 53.0391% test。

## 数据来源

六个数据集为真实复制文件，不是指向其他工程的链接。优先使用 baseline 中仍存在的文件，缺少项从
`QDRL-unified-research/QDRL/data` 补齐。复制前核对两处现存重叠文件，SHA256 全部相同；
逐文件来源和 SHA256 见 DATA_MANIFEST.json。WIKI 没有静态图，不替代或伪造该资源。

## 保留边界

实体任务、TF、状态 on/input、独立 C2 关闭、长期槽关闭是主配置。state-off/global/C2 开关只供显式对照。
原始时间边界、每候选 null、共享单次 K/V、双端读出、窗口内状态重算都保持 R02 语义。
共享的主干嵌入别名属于原工程的参数关系；checkpoint 使用 strict load 和别名验证。

本次不执行正式训练或真实 test，也不生成其他数据集的 MRR 成果。
