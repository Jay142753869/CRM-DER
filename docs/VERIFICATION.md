# CRM-DER 交付检查

日期：2026-10-16。环境：Conda qdrl，Python 3.8.20，PyTorch 1.13.1，DGL 1.1.2+cu117，numpy 1.24.4。

## 已完成

- `python -B -m unittest discover -s tests -v`：11 项通过，见 TEST_GREEN.log。新增按数据集窗口、未核实窗口拒绝隐式训练、checkpoint 继承检查。
- 历史窗口证据复核：实测六个数据集的全量快照数与相邻快照间隔，与 `--time-unit` 声明逐项吻合；对论文 Fig. 2 做像素级复测（`history_window_evidence/measure_fig2_peaks.py`），四个子图各自独立反推出同一组窗口网格 `{1, 3, 7, 14, 30, 60}`，峰值位置 (b) = 14、(c) = 7、(d) = 7 独立支持 §5.6 的笔误修正。结论与依据见 HISTORY_WINDOWS.md。
- `python -B verify_project.py --smoke`：六个数据集文件 SHA256、字典轴、train/valid 加载，以及真实完整候选轴 CPU 前向/反向通过，见 DATA_SMOKE_RESULTS.json。
- 实际 subprocess 启动 main.py：默认 `./data` 加载、`models/<run-name>/best.pt` 保存、`--test --load-model`、CSV 输出通过。
- 合成数据连续两轮与训练一轮再恢复一轮的模型、Adam 状态及 Torch CPU RNG 逐位相同，scheduler 状态相同。
- 显式更改 state-mode 的 resume 被拒绝；同名新训练不覆盖已有 checkpoint。
- 最小 test 命令正确恢复静态开关后再核对数据身份，不需要手工重复训练模型参数。
- 极端 logits 的稳定 NLL 和梯度有限，避免 `log(softmax)` 下溢。
- 所有 Python 文件语法检查通过；运行源码没有指向原 report 或其他工程的绝对路径依赖。

## 来源保持

原 R02 的 23 个复制文件中，19 个逐字节相同。改动集中在：

1. opt.py：本地路径、命令别名、用户选定的 baseline 默认协议及参数校验。
2. src/backbone/model.py：仅 task_loss 的 log_softmax 数值实现。
3. src/runtime.py：scheduler 状态持久化与新增源码的恢复身份检查。
4. src/trainer.py：训练顺序/损失/早停对齐、完整恢复、进度输出及结果记录。

src/crm/core.py、model.py、factory.py、frame_groups.py 与全部推理评分模块保持原来源。
新增 src/experiment.py 只管理参数恢复、输出目录与 CSV，不参与模型评分。
SOURCE_MANIFEST.json 保存复制前的来源，PACKAGED_SOURCE_SHA256.json 保存交付源码的最终哈希。
后者改由 `docs/build_packaged_hashes.py` 生成（带 `--check`）。原先手工维护的清单在
02:27 生成后，02:31 又改过 opt.py 与 tests/test_standalone.py 而没有重新生成，
曾与工作树不符；现已重建并通过 `--check`（26 条全部匹配）。

本轮历史窗口复核只改动文档（README.md、docs/HISTORY_WINDOWS.md、docs/VERIFICATION.md），
新增 docs/history_window_evidence/ 的复测脚本与产物、docs/build_packaged_hashes.py。
main.py、opt.py、verify_project.py 与 src/ 下源码均未改动。

## 审查结果

正确性：通过参数、数值、真实数据装配与训练生命周期检查，修复了旧 R02 的 test 静态身份恢复顺序问题。
可读性与架构：模型代码和运行/输出适配分开，保留 R02 结构及明确模型名称。
文件与输入处理：独立复制数据、显式可信 checkpoint、身份核对、输出目录覆盖保护。
性能：保留原候选与静态分块重算；没有增加新模型头、候选裁剪或新运行依赖。
当前交付范围内没有待修复的阻断问题。

## 未执行

没有进行其他数据集正式维度及其对应历史窗口的 GPU 训练、完整真实 test、额外适应或性能比较。
真实数据 smoke 使用 D32/H3、CPU、完整单帧候选轴，只做主任务梯度检查；静态监督和 optimizer 闭环另由合成数据回归覆盖。
因此没有其他数据集的正式 MRR，也不承诺其完整规模训练都能装入当前显存。
原 QDRL-main-new 与原 R02 工程均未修改。
