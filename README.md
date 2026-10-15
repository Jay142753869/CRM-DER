# CRM-DER

CRM-DER（R02）是“统一因果关系事件记忆 + 双端条件读出”的独立时序知识图谱实体预测工程。
本目录中的 `main.py` 实际构造 R02 的 CRMModel；运行名称、输出和工程名称均为 CRM-DER。
模型保留 R02 的主干、共享 K/V、输入门控状态槽和双端读出，独立 C2、长期槽默认关闭。

按用户选择，本工程默认采用 `QDRL-main-new/QDRL` 的训练方式：时间顺序遍历训练快照，
每个快照正反向分别优化，Adam + 固定静态图损失权重 + 完整 valid 选模，早停容忍 10 轮。
这与历史 R02 已测训练协议不同，新实验不能直接当作原 seed43 结果复现。
具体差异与证据见 [docs/TRAINING_ALIGNMENT.md](docs/TRAINING_ALIGNMENT.md)。

## 目录

```text
CRM-DER/
├── main.py                 # 训练、恢复、显式 test 入口
├── opt.py                  # baseline 风格参数 + CRM 专属参数
├── src/
│   ├── crm/                # R02 因果状态、事件 bank、双端读出
│   ├── backbone/           # R02 保留的动态图、静态图、MA、查询与解码器
│   ├── c2/                 # 显式恢复对照；默认关闭
│   ├── data.py
│   ├── trainer.py
│   ├── runtime.py
│   ├── evaluation.py
│   └── experiment.py
├── data/                   # 六个数据集的独立文件副本
├── models/<run-name>/      # best.pt、last.pt、config.json、epoch_metrics.jsonl、summary.json
├── result/                 # <DATASET>.csv 及各次显式 test 的 JSON
├── tests/                  # 参数、数值、数据及训练/恢复/test 回归检查
├── docs/                   # 来源哈希、协议差异与检查结果
└── verify_project.py
```

所有命令均在本目录运行，使用已有 Conda 环境 `qdrl`。本工程不依赖原 report 中的源码路径，
数据也无需指向其他实验副本；可以整目录搬走使用。

## 训练

```powershell
conda activate qdrl
cd D:\improve-QDRL\CRM-DER
python main.py -d ICEWS18 --train-history-len 7 --lr 0.0005 --n-layers 2 --gpu 0 --n-hidden 256 --decoder convtranse --encoder uvrgcn --layer-norm --pre-weight 0.7 --pre-type TF --add-static-graph --add_query --seed 42 --evaluate-every 1 --test-every 0 --n-epochs 50 --run-name ICEWS18_seed42 --report-subgroups
```

上面的 ICEWS18 窗口为 7。ICEWS14、YAGO 也使用 7；**ICEWS05-15 改为 `--train-history-len 14`**。
省略该参数时，程序按下表选择有论文图示依据的窗口，不再对所有数据集统一使用 7。
这些是参照 QDRL 的起始设置，尚未证明是 CRM-DER 的最优值。

**GDELT、WIKI 尚无已核实的论文窗口，首次训练必须显式指定**。
WIKI 没有静态图；下面让使用者先输入已确定的实验窗口，再启动：

```powershell
$historyLen = [int](Read-Host '输入本次 WIKI 实验确定的历史快照数')
python main.py -d WIKI --train-history-len $historyLen --lr 0.0005 --n-layers 2 --gpu 0 --n-hidden 256 --decoder convtranse --encoder uvrgcn --layer-norm --pre-weight 0.7 --pre-type TF --add_query --seed 42 --evaluate-every 1 --test-every 0 --n-epochs 50 --run-name WIKI_seed42 --report-subgroups
```

`--run-name` 可省略，此时自动生成包含数据集、seed 和时间的名称。已有训练输出不会被新的同名训练覆盖。
可使用 `--run-dir` 自定权重目录。权重为 `models/<run-name>/best.pt` 和 `last.pt`，不要求预先存在 baseline 权重。
默认从头训练；`--common-state` 仅用于显式配对初始化，绝不能把其他数据集的权重当作公共初始化。

## 测试与恢复

训练中只读取 train/valid，按 valid 的整体 filtered MRR 保存 best。训练结束后显式测 test：

```powershell
python main.py -d ICEWS18 --gpu 0 --test --load-model .\models\ICEWS18_seed42\best.pt --report-subgroups
```

模型尺寸、静态图开关、历史窗口等从 checkpoint 恢复，不必重写训练参数；`-d` 必须与权重数据集一致。
输出打印整体、正向、逆向的 filtered 指标，以及整体 raw 指标；添加 `--report-subgroups` 可报告 R/A/B。
CSV 保存在 `result/ICEWS18.csv`，数值为 0–1，`filter_all_MRR` 对应双向整体 filtered MRR。
每次显式 test 的 config/test JSON 放在 `result/<run-name>-test-<time>/`。

从 last 接着训练：

```powershell
python main.py -d ICEWS18 --gpu 0 --resume --load-model .\models\ICEWS18_seed42\last.pt --n-epochs 50
```

恢复包含模型、Adam、可选 scheduler、随机数状态、快照顺序和正反向游标。
`--n-epochs` 是目标总轮数；若已经完成早停，恢复不会自动延长训练。
显式更改机制、优化设置或数据身份将被拒绝。源码修改也会触发恢复身份检查。
checkpoint 只能加载本项目或可信的 R02 文件。

## 数据集、历史窗口与时间单位

| 数据集 | 实体数 | 原始关系数 | 静态图 | 默认历史快照数 | time-unit |
|---|---:|---:|---|---|---:|
| ICEWS14 | 7128 | 230 | 有 | 7（§5.6） | 1 |
| ICEWS18 | 23033 | 256 | 有 | 7（§5.6、Fig. 2c） | 24 |
| ICEWS05-15 | 10488 | 251 | 有 | 14（§5.6 笔误修正、Fig. 2b） | 1 |
| GDELT | 7691 | 240 | 有 | 未核实，必须显式指定 | 15 |
| YAGO | 10623 | 10 | 有 | 7（§5.6、Fig. 2d） | 1 |
| WIKI | 12554 | 24 | 无 | 论文未实验，必须显式指定 | 1 |

论文正文 §5.6 的第二个 `ICEWS18` 是 `ICEWS05-15` 的笔误，窗口依据、Fig. 2 的
像素级复测、图像和限制见 [docs/HISTORY_WINDOWS.md](docs/HISTORY_WINDOWS.md)。
`--train-history-len` 是取多少个历史快照；`--time-unit` 只归一化年龄特征，两者不能混用。
时间单位沿用 R02 显式元数据表，历史按指定数量的真实快照选取，因果边界始终为原始 `t<T`。
新增数据集可放到 `data/<NAME>`，提供 train/valid/test、entity2id、relation2id、stat；
静态模式另外需要正确的 `e-w-graph-llm.txt`。可用 `--time-unit` 显式指定年龄归一化单位。

## 训练默认值与选项

| 选项 | 默认 |
|---|---|
| seed / gpu | 42 / 0 |
| D / 历史窗口 / 图层数 | 256 / 按数据集（见上表） / 2 |
| 学习率 / weight decay / dropout / clip | 0.0005 / 1e-5 / 0.2 / 1 |
| 训练顺序 | chronological |
| 静态图 | 需要显式 `--add-static-graph`；WIKI 不加 |
| 静态损失 | `main + 0.1 * static`；仅正向有静态三元组监督 |
| 选模 / 早停 | 每轮完整 valid，整体 filtered MRR，连续 10 轮下降 |
| scheduler | none；可选 `--lr-scheduler plateau` |
| CRM 状态 / 门控 | on / input，状态秩 8×32，读出 hidden64 |
| C2 / 长期槽 | off / off |

可调整 `--static-loss-weight`、`--static-warmup-epochs`；`--static-weight` 是同义参数。
可选 `--state-mode off`、`--state-gate global`、`--enable-c2` 做独立消融。
`--add_query` 与 `--add-query` 同义，`--load-model` 与 `--checkpoint` 同义。
关系预测、GRU、外部预处理历史图与长期槽未由 R02 实现，不会静默改成其他模型。

## 检查

```powershell
python -B -m unittest discover -s tests -v
python -B verify_project.py --smoke
```

第一项使用合成小数据验证完整训练/恢复/test 闭环，并读取六个真实数据集 train/valid 核对数据轴。
第二项检查数据文件哈希，并在每个真实数据集上用 D32/H3 做完整候选轴的 CPU 前向/反向；
没有创建优化器、不读取真实 test 标签、不生成研究指标。
这些检查不代替各数据集 D256/H7 的正式 GPU 训练与资源验证。
