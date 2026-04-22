# 实验 1：CM2026-Project1-星光修复师

在遥远的「光栅王国」里，所有图像都不是由像素直接绘制出来的，而是由许多漂浮在空中的星光微粒组成。每个微粒对应一个可学习的 2D Gaussian，拥有位置、尺度、颜色，以及可选的透明度。修复师们无法直接逐像素描摹它，只能召唤有限数量的星光微粒，让这些微粒自动移动、调整大小与颜色，最终重新拼出原图。你的目标，就是不断调整这些微粒的参数，使生成图像尽可能接近目标图像。

优化问题的详细介绍见[docs/problem_formulation.md](/home/lkj/code/cm_pr1/minimal_2dgs/docs/problem_formulation.md:1)

这份仓库是课程实验的 starter code。它提供了可运行基线，也预留了 loss、初始化器、优化器、调度器和目标图生成器的扩展接口，方便做消融实验和竞赛。

优化过程中，高斯微粒逐步移动、缩放、变色，逼近目标图像：

<p align="center">
  <img src="docs/optimization_progress.png" width="720" alt="Optimization progress">
  <br>
  <em>从左到右：第 50 / 100 / 150 / 200 步的重建结果</em>
</p>

## 快速开始

下图展示了基线配置（100 个高斯，MSE loss，Adam 优化器）在 200 步训练后的效果：

<p align="center">
  <img src="docs/comparison_baseline.png" width="720" alt="Baseline comparison">
  <br>
  <em>左：目标图像 | 中：高斯重建 | 右：绝对误差</em>
</p>

安装依赖：

```bash
pip install -r requirements.txt
```

运行默认训练：

```bash
python main.py
```

也可以只生成当前配置对应的目标图像。这个脚本既支持真实图片目标，也支持 txt 高斯目标：

```bash
python generate_target.py
```

## 实验要求

本实验的目标是：给定一张目标图像，用一组可学习的 2D Gaussians 去重建它，并比较不同设计对优化效果的影响。分为**消融**和**竞赛**两个模块。

在**消融**模块你需要完成两类工作：

- 补全指定模块，例如 loss、初始化器、优化器和调度器。
- 基于统一基线完成消融实验，并进一步组合自己的方案参加竞赛。

建议顺序：

1. 先跑通默认基线。
2. 补全要求实现的源码文件。
3. 按模块分别做消融实验。
4. 最后整理自己的最佳配置。

在**竞赛**模块你需要对两种竞赛配置进行相应的调参，也可以自定义一些模块以提升优化的最终效果。

评分构成：

| 部分 | 占比 |
| ---- | ---- |
| 消融 A：Loss 函数 | 12% |
| 消融 B：初始化策略 | 12% |
| 消融 C：优化器 | 12% |
| 消融 D：模型设计 | 12% |
| 消融 E：学习率调度器 | 12% |
| 竞赛 Sprint | 20% |
| 竞赛 Standard | 20% |
| 合计 | 100% |

最低必做项：

| 类型 | 文件 |
| ---- | ---- |
| 必做 | [losses.py](/home/lkj/code/cm_pr1/minimal_2dgs/losses.py:1) |
| 必做 | [schedulers.py](/home/lkj/code/cm_pr1/minimal_2dgs/schedulers.py:1) |
| 必做 | [optimizers/student_sgd.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_sgd.py:1) |
| 必做 | [optimizers/student_momentum.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_momentum.py:1) |
| 必做 | [optimizers/student_adam.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_adam.py:1) |
| 必做 | [initializers/image_sample_init.py](/home/lkj/code/cm_pr1/minimal_2dgs/initializers/image_sample_init.py:1) 或 [initializers/bright_spot_init.py](/home/lkj/code/cm_pr1/minimal_2dgs/initializers/bright_spot_init.py:1) |
| 选做 | [optimizers/student_adamw.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_adamw.py:1) |
| 选做 | [optimizers/student_muon.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_muon.py:1) |
| 选做 | [optimizers/student_newton.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_newton.py:1) |

建议先完成所有必做项，再开始竞赛调参。

建议提交内容：

| 内容 | 说明 |
| ---- | ---- |
| 代码 | 你补全或新增的源码文件 |
| 结果表 | 至少包含 `PSNR / MSE / MAE` |
| 可视化 | loss 曲线、重建图、误差图中的若干项 |
| 分析 | 简短说明你的方法与观察结论 |

## 你主要会改哪里

所有主要配置都在 [config.py](/home/lkj/code/cm_pr1/minimal_2dgs/config.py:1)。

最常改的配置项：

- `config.target`：目标图来源、图像尺寸、txt 路径
- `config.model`：高斯数量、是否各向异性、是否启用 alpha
- `config.initializer`：初始化策略
- `config.loss`：loss 类型与权重
- `config.optimizer`：优化器类型与超参数
- `config.scheduler`：学习率调度器
- `config.train`：训练步数、打印频率、保存频率

除了改配置，学生还需要补全一些作业文件。常见入口如下：

| 模块 | 需要改的文件 | 说明 |
| ---- | ---- | ---- |
| Loss | [losses.py](/home/lkj/code/cm_pr1/minimal_2dgs/losses.py:1) | 实现 `l1`、`charbonnier`、`mse_l1`、`mse_edge` 等 loss |
| Scheduler | [schedulers.py](/home/lkj/code/cm_pr1/minimal_2dgs/schedulers.py:1) | 实现 `cosine`、`warmup_cosine`、`step_decay` |
| Optimizer | [optimizers/student_sgd.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_sgd.py:1) | 实现 SGD |
| Optimizer | [optimizers/student_momentum.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_momentum.py:1) | 实现 SGD + Momentum |
| Optimizer | [optimizers/student_adam.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_adam.py:1) | 实现 Adam |
| Optimizer Bonus | [optimizers/student_adamw.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_adamw.py:1) | 选做：实现 AdamW |
| Optimizer Bonus | [optimizers/student_muon.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_muon.py:1) | 选做：实现 Muon |
| Optimizer Bonus | [optimizers/student_newton.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_newton.py:1) | 选做：实现 Newton 风格方法 |
| Initializer | [initializers/image_sample_init.py](/home/lkj/code/cm_pr1/minimal_2dgs/initializers/image_sample_init.py:1) | 图像采样初始化 |
| Initializer | [initializers/bright_spot_init.py](/home/lkj/code/cm_pr1/minimal_2dgs/initializers/bright_spot_init.py:1) | 亮度感知初始化 |

如果你想自己扩展新模块，也可以参考这些模板：

- [optimizers/custom_optimizer_template.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/custom_optimizer_template.py:1)
- [initializers/custom_initializer_template.py](/home/lkj/code/cm_pr1/minimal_2dgs/initializers/custom_initializer_template.py:1)
- [target_generators/custom_target_generator_template.py](/home/lkj/code/cm_pr1/minimal_2dgs/target_generators/custom_target_generator_template.py:1)

## 训练输出

训练结束后，默认会在 `outputs/` 下保存：

- `target.png`
- `reconstruction_final.png`
- `comparison.png`
- `loss_curve.png`
- `metrics.txt`
- 若干 `recon_step_*.png`

如果 `config.visualization.save_video = True`，还会额外导出优化过程动画。

## 文档导航

- [docs/problem_formulation.md](/home/lkj/code/cm_pr1/minimal_2dgs/docs/problem_formulation.md:1)
  优化问题形式化
- [docs/ablation_experiments.md](/home/lkj/code/cm_pr1/minimal_2dgs/docs/ablation_experiments.md:1)
  消融实验要求
- [docs/competition.md](/home/lkj/code/cm_pr1/minimal_2dgs/docs/competition.md:1)
  竞赛规则与提交方式
- [data/README.md](/home/lkj/code/cm_pr1/minimal_2dgs/data/README.md:1)
  数据说明

## 代码结构

- `main.py`：训练入口
- `train.py`：主训练循环
- `config.py`：结构化配置
- `models/`：高斯参数模型
- `renderer/`：可微渲染器
- `losses.py`：loss 与工厂
- `optimizers/`：优化器
- `initializers/`：初始化器
- `schedulers.py`：学习率调度器
- `target_generators/`：目标图生成器
- `experiments/`：竞赛脚本与配置模板

## 补充说明
- 可选项内容如果不完成不会被特别扣分，但如果提交了可选项的内容可视情况提高原本分数。
- 竞赛的评分细则仅供参考，可能发生变化。由于竞赛是开放设定，没有标准答案，鼓励使用有趣的优化。

## 学术诚信

请独立完成本次作业。

- 可以阅读课程提供的 starter code、文档和参考资料。
- 可以与同学讨论思路，但不要直接交换代码、实验结果或报告文本。
- 不要抄袭他人实现，也不要将他人的代码改名后作为自己的提交。
- 如果使用了课外资料、工具或生成式 AI，请遵守课程要求，并在提交中如实说明其用途。
- 提交的代码、实验结果和分析应当与自己的实际实现一致，不得作弊。
- 允许使用AI工具完成本次作业，但需要确保你知道AI在做什么。

Have Fun! 
