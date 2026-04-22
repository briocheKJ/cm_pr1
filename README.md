# 实验 1：CM2026-Project1-星光修复师

在遥远的「光栅王国」里，所有图像都不是由像素直接绘制出来的，而是由许多漂浮在空中的星光微粒组成。每个微粒对应一个可学习的 2D Gaussian，拥有位置、半径、颜色，以及透明度。修复师们无法直接逐像素描摹它，只能召唤有限数量的星光微粒，让这些微粒自动移动、调整大小与颜色，最终重新拼出原图。你的目标，就是不断调整这些微粒的参数，使生成图像尽可能接近目标图像。优化问题的详细介绍见[docs/problem_formulation.md](/home/lkj/code/cm_pr1/minimal_2dgs/docs/problem_formulation.md:1)

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

只生成目标图像：

```bash
python generate_target.py
```

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
| Optimizer Bonus | [optimizers/student_adamw.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_adamw.py:1) | 可选实现 AdamW |
| Optimizer Bonus | [optimizers/student_muon.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_muon.py:1) | 可选实现 Muon |
| Optimizer Bonus | [optimizers/student_newton.py](/home/lkj/code/cm_pr1/minimal_2dgs/optimizers/student_newton.py:1) | 可选实现 Newton 风格方法 |
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
