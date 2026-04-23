# 最终测试说明

## 目标

自由组合模块和超参数，在测试集上达到尽可能高的平均 PSNR。

建议先完成必做模块和消融实验，再进入最终测试部分。最终测试不采用同学之间的排名，而是按固定 PSNR 阈值给分。

## 两个赛道

| 赛道 | 训练步数 | 重点 |
| ---- | ---- | ---- |
| Sprint | `100` | 快速收敛 |
| Standard | `500` | 最终质量 |

## 硬约束

以下设置不可修改：

| 项目 | 约束 |
| ---- | ---- |
| 高斯数量 | `200` |
| 背景色 | `(0.0, 0.0, 0.0)` |
| 随机种子 | `42` |
| 图像大小 | `128 x 128` |

## 可以调的内容

| 模块 | 可调内容 |
| ---- | ---- |
| Loss | 类型与超参数 |
| Initializer | 类型与超参数 |
| Optimizer | 类型与超参数 |
| Scheduler | 类型与超参数 |
| Param Groups | 各参数组 lr 倍率 |
| Model | 是否各向异性、是否启用 alpha |

## 测试集

最终测试使用10张测试图像，包括5张真实图像和5张合成图像。最终成绩取10张图的平均PSNR。

真实图像示意：

<p align="center">
  <img src="competition_real_images.png" width="720" alt="Competition real images">
</p>

合成目标示意：

<p align="center">
  <img src="competition_synthetic_images.png" width="720" alt="Competition synthetic images">
</p>

## 评分

每个赛道单独按平均 PSNR 给分，满分 20 分。低于最低阈值不得分。

默认实现的平均 PSNR 大约为 Sprint `23.1 dB`、Standard `27.1 dB`。因此最低得分线设置在默认实现之上，直接提交默认配置不能获得最终测试分数。

### Sprint 赛道

| 平均 PSNR | 得分 |
| ---- | ---- |
| `>= 26.5 dB` | 20 |
| `>= 26 dB` | 16 |
| `>= 25 dB` | 12 |
| `>= 23.5 dB` | 8 |
| `< 23.5 dB` | 0 |

### Standard 赛道

| 平均 PSNR | 得分 |
| ---- | ---- |
| `>= 30 dB` | 20 |
| `>= 29 dB` | 16 |
| `>= 28 dB` | 12 |
| `>= 27.5 dB` | 8 |
| `< 27.5 dB` | 0 |

总分 = Sprint + Standard，满分 40。

## 验收内容

竞赛部分验收时，至少需要看到以下内容：

- `competition_settings.py`：必交，且其中需要给出 `get_sprint_setting()` 和 `get_standard_setting()`。
- 报告中的竞赛结果：至少包含 Sprint 和 Standard 两个赛道的结果。
- 简短说明：说明你最终采用了哪些设置，以及为什么这样设计。

如果你只做配置搜索，没有改动算法实现，那么只需提交 `competition_settings.py`。如果你修改了优化器、初始化器、loss 或 scheduler 的实现，那么除了 `competition_settings.py`，还需要提交所有改过的 `student/*.py` 文件。

补充说明：

- 如果你新增了优化器、初始化器、loss 或 scheduler，需要在对应的 `build_*` 函数中完成注册。
- 不需要单独提交竞赛运行产生的 `outputs/` 结果文件，结果图表和数值直接整理进报告即可。
