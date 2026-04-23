# 消融实验说明

## 实验原则

每次只改变一个模块，其余设置保持默认基线不变。如果需要改变默认基线，请在实验报告中说明并阐明原因。单个消融实验需保证除消融模块其他模块配置一致。

## 默认基线

| 项目 | 默认设置 |
| ---- | ---- |
| 图像 | `data/real_images/Starry_Night_256.png` |
| 图像大小 | `256 x 256` |
| 高斯数量 | `100` |
| 步数 | `200` |
| Loss | `mse` |
| Optimizer | `torch_adam` |
| Scheduler | `constant` |
| Initializer | `random` |
| 各向异性 | `True` |
| Alpha | `True` |
| 随机种子 | `42` |


## 验收内容

所有消融实验在报告中至少需要包含以下内容：

- 结果表格：至少汇报 `PSNR / MSE / MAE`。
- 可视化结果：至少给出 `loss curve` 或最终重建对比图。
- 简短分析：每部分建议不超过 300 字，说明方法变化带来的影响。

代码提交时，需要至少提交以下 4 个文件：

- `student/losses.py`
- `student/optimizers.py`
- `student/initializers.py`
- `student/schedulers.py`

补充说明：

- 即使某一部分实验没有完全做完，也需要提交这 4 个文件，便于统一验收。
- 如果你在这 4 个文件之外还新增了辅助实现，也需要一并提交。
- 不需要单独提交 `outputs/` 下的结果文件，图表和数值直接整理进实验报告即可。


## 五个消融实验总表

共性要求：

- 每次只改一个模块
- 其余设置保持默认基线
- 统一汇报 `PSNR / MSE / MAE`

| 实验 | 模块 | 需要做什么 | 比较对象 | 其余保持不变 | 重点观察 | 补充说明 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| A | Loss | 在 `student/losses.py` 中实现至少 2 种新 loss | `mse`, `l1`, `mse_l1` | 初始化、优化器、调度器、模型开关 | 最终 PSNR、曲线平滑性、收敛速度 | 无 |
| B | 初始化策略 | 在 `student/initializers.py` 中实现至少 1 种图像感知初始化 | `random` vs 你的方法 | loss、优化器、调度器、模型开关 | 前 50 步收敛速度、最终 PSNR | 重点看前期收敛 |
| C | 优化器 | 实现 `student_sgd`, `student_momentum`, `student_adam`，`student_adamw`, `student_muon`(可选) | `torch_adam`, `student_sgd`, `student_momentum`, `student_adam`，`student_adamw` | 初始化、loss、调度器、模型开关 | 收敛速度、稳定性、最终 PSNR | `student_adam` 应与 `torch_adam` 接近 |
| D | 模型设计 | 无需实现新代码，仅切换配置 | 见下方 4 种组合 | 初始化、loss、优化器、调度器 | 各向异性和 alpha 对 PSNR 的贡献 | 比较开关是否互补 |
| E | 学习率调度器 | 在 `student/schedulers.py` 中实现至少 1 种调度器 | `constant`, `cosine`, `warmup_cosine`, `step_decay` | 初始化、loss、优化器、模型开关 | 中后期收敛行为、最终 PSNR | 重点看后期是否更稳 |

## 实验 D 的 4 种组合

| 编号 | `use_anisotropic` | `use_alpha` | 说明 |
| ---- | ---- | ---- | ---- |
| D1 | `False` | `False` | 各向同性 + 无 alpha |
| D2 | `True` | `False` | 各向异性 + 无 alpha |
| D3 | `False` | `True` | 各向同性 + alpha |
| D4 | `True` | `True` | 各向异性 + alpha |


## 推荐的结果整理方式

| 图表 | 用途 |
| ---- | ---- |
| 总表格 | 汇总每组 `PSNR / MSE / MAE` |
| Loss 曲线图 | 比较收敛速度与稳定性 |
| 重建对比图 | 比较视觉质量 |
| 误差图 | 比较残差分布 |
