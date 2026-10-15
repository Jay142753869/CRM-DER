# 按数据集核对历史窗口（2026-10-16 复核）

此前交付将全部数据集统一写为 `--train-history-len 7`，依据不足，现更正。
本次核对本地 QDRL 论文 PDF、作者 README、原始 opt.py 及本地训练记录，
并对论文 Fig. 2 做像素级复测（脚本与产物见
[history_window_evidence/](history_window_evidence/)）。

## 有据可用的起始设置

| 数据集 | 本工程默认 | 依据 |
|---|---:|---|
| ICEWS14 | 7 | §5.6 正文直述；本地 baseline 正式命令 |
| ICEWS05-15 | 14 | §5.6 正文（原文此处误写为 ICEWS18）；Fig. 2(b) 峰值复测 = 14 |
| ICEWS18 | 7 | §5.6 正文直述；Fig. 2(c) 峰值复测 = 7 |
| YAGO | 7 | §5.6 正文直述；Fig. 2(d) 峰值复测 = 7 |
| GDELT | 不自动填值 | 论文没有该数据集的窗口曲线，未查到专属正式命令 |
| WIKI | 不自动填值 | 不在这篇论文的实验数据集内 |

这张表给出参照 QDRL 的起始配置，不证明 CRM-DER 在各数据集上的最优窗口。
图示没有附原始数值表，尤其不能把接近平台的微小差异解释成显著优势。

## 原文有笔误，不能只抄一句话

本地论文 [QDRL.pdf](../../report/QDRL.pdf)（在本工程目录之外，整目录搬走后该链接会失效），
PDF 第 15 页（印刷页 421），§5.6 原文：

> The optimal results are achieved when m = 7 on ICEWS14, ICEWS18, and YAGO,
> and when m = 14 on ICEWS18, suggesting that a moderate historical context is
> sufficient for most datasets.

第二个 `ICEWS18` 与第一个重复。Fig. 2(b) 中 ICEWS05-15 的 MRR 峰值明确落在
m = 14，且该图没有给出"ICEWS18 最优为 14"的任何迹象（见下节复测），
因此判定第二个 `ICEWS18` 是 ICEWS05-15 的笔误。修正后正文给出
`ICEWS14 = 7 / ICEWS05-15 = 14 / ICEWS18 = 7 / YAGO = 7`，与本工程默认表一致。

核对 PDF 第 14 页（印刷页 420）Fig. 2，(b) 对应 ICEWS05-15、(c) 对应 ICEWS18。
原始页面已渲染保存为
[history_window_evidence/QDRL_page14.png](history_window_evidence/QDRL_page14.png)。

## Fig. 2 的窗口网格与峰值复测

`history_window_evidence/measure_fig2_peaks.py` 从渲染图里定位 MRR 曲线
（tab10 蓝）的实心圆数据点，不做绝对像素定标，改用相邻间距的比例反推网格值
（锚定依据是正文写明的 m = 1、m = 3、m ≥ 30，以及 x 轴上限恰为 60）。

四个子图各自独立反推，全部收敛到同一组整数网格，最大偏差 0.08 个窗口单位：

```text
m ∈ {1, 3, 7, 14, 30, 60}
```

**7 与 14 都落在论文实测的网格点上，不是插值出来的中间值。**

复测到的 MRR 峰值位置：

| 子图 | 复测峰值 | 与次高点的像素差 | 与正文的关系 |
|---|---:|---:|---|
| (a) ICEWS14 | m = 30 | 1.5 px | 7 / 14 / 30 处于平台内，取正文的 7 |
| (b) ICEWS05-15 | m = 14 | 1.5 px | 与正文（笔误修正后）一致 |
| (c) ICEWS18 | m = 7 | 0.5 px | 与正文一致 |
| (d) YAGO | m = 7 | 2.0 px | 与正文一致 |

(b)(c)(d) 三张子图的峰值位置独立支持正文的取值，也就是独立支持
"第二个 ICEWS18 是笔误"这一判断。

(a) 的复测峰值落在 30 而不是 7：三个点的纵向像素差都在 1.5 px 以内，
在图中不可分辨，属于平台噪声，不足以推翻正文的 m = 7。此处如实记录，
避免以后把平台内的微小差异当作"图示支持 7"来引用。

复测产物：`fig2_peak_measurement.json`（逐点坐标与网格反推）、
`fig2_peaks_annotated.png`（圈出检测点并标出峰值）。

## 数据层面的可行性

六个数据集的快照数与相邻快照间隔（实测，六个数据集全量时间戳）：

| 数据集 | 快照数 | 相邻快照间隔 | 声明的 time-unit |
|---|---:|---:|---:|
| ICEWS14 | 365 | 1 | 1 |
| ICEWS18 | 304 | 24 | 24 |
| ICEWS05-15 | 4017 | 1 | 1 |
| GDELT | 2751 | 15（另有缺槽 30 / 45 / 60） | 15 |
| YAGO | 189 | 1 | 1 |
| WIKI | 232 | 1 | 1 |

`--train-history-len` 取的是快照个数，H7 / H14 在六个数据集上都远小于可用快照数，
不存在窗口大于历史深度的情况。`--time-unit` 只归一化年龄特征，不是历史窗口
（例如 ICEWS18 的 time-unit = 24，不代表看 24 个历史快照）；声明值与实测间隔逐项吻合。

评估阶段的候选历史取自 train + valid 的合并帧序（`src/trainer.py` 的
`Runtime.eval_frames`），所以 YAGO valid 只有 5 帧也不会出现历史为空；
训练阶段只用 `train_frames`，不会看到 valid 快照。

## 其余依据的强度

作者公开 [README](https://github.com/vzhang-mm/QDRL#training) 给的是
`<DATASET> --train-history-len 7` 通用示例，不含逐数据集配置表，
不能证明所有数据集正式实验都用 7（本次已联网复核该 README，确认无配置表）。
本地原始 `QDRL-main_backup/QDRL/opt.py` 的通用默认值为 10，
同样不等于 GDELT 或 WIKI 的专属最优值。
本地现存 baseline 日志只找到 ICEWS14 的 H7，不能用其他模型（如 TiRGN）的窗口冒充。

## 程序行为

- 已有证据的四个数据集：省略 `--train-history-len` 时使用上表；显式传入值始终优先。
- GDELT、WIKI、新增数据集：首次训练必须传 `--train-history-len`，没有悄悄回退到 7 或 10。
- `--test` / `--resume`：允许省略窗口，由 checkpoint 中实际保存的设置恢复；新默认不会覆盖旧权重。

原先六数据集 CPU smoke 明确使用 D32/H3，仅验证工程装配，不能用其 H3 作为正式训练设置。
本次没有运行窗口搜索，没有读真实 test 来选择窗口，也没有改动 baseline 或原始 R02 的历史记录。
