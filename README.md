# 2D 高斯溅射图像拟合 — 计算方法课程设计

本项目是一个面向本科生的 2D Gaussian Splatting 图像拟合教学框架。给定一张目标图像，系统使用一组可微分的 2D 高斯函数逐步逼近它。

## 快速开始

```bash
pip install -r requirements.txt
python main.py                # 运行基线（mse + torch_adam + random init）
```

输出保存在 `outputs/` 下，重点查看：
- `metrics.txt`：最终 PSNR / MSE / MAE
- `comparison.png`：目标 / 预测 / 误差对比
- `loss_curve.png`：训练损失曲线

## 目录结构

```
minimal_2dgs/
├── main.py                     # 入口，支持 --mode student/teacher
├── config.py                   # 所有可配置参数
├── train.py                    # 训练循环
├── losses.py                   # [学生文件] Loss 函数
├── schedulers.py               # [学生文件] 学习率调度器
├── mode.py                     # student/teacher 模式切换
├── evaluation.py               # 评估指标计算
├── utils.py                    # 工具函数
├── models/
│   └── gaussian_model.py       # 2D 高斯参数模型，含参数分组接口
├── renderer/
│   └── gaussian_renderer.py    # 可微渲染器
├── optimizers/
│   ├── factory.py              # 优化器工厂
│   ├── torch_baselines.py      # PyTorch Adam（基线）
│   ├── student_sgd.py          # [学生文件] SGD
│   ├── student_momentum.py     # [学生文件] SGD + Momentum
│   ├── student_adam.py         # [学生文件] Adam
│   ├── student_adamw.py        # [学生文件] AdamW
│   ├── student_muon.py         # [学生文件] Muon（bonus）
│   └── student_newton.py       # [学生文件] Newton（bonus）
├── initializers/
│   ├── factory.py              # 初始化工厂
│   ├── random_init.py          # 随机初始化（基线）
│   ├── grid_init.py            # 网格初始化
│   ├── image_sample_init.py    # [学生文件] 图像采样初始化
│   └── bright_spot_init.py     # [学生文件] 亮度感知初始化
├── target_generators/          # 目标图像生成器
├── experiments/
│   ├── experiment1_spec.md     # 实验说明书
│   ├── run_competition.py      # 竞赛评测脚本
│   └── my_competition_config.py # 竞赛提交模板
└── data/
    ├── Starry_Night_256.png    # 默认目标图像
    ├── examples/               # txt 格式高斯目标示例
    └── competition/            # 竞赛测试图像
```

## 基线配置

| 模块     | 基线选项                       |
| -------- | ------------------------------ |
| Loss     | `mse`                          |
| Optimizer| `torch_adam`（lr=0.05）         |
| Scheduler| `constant`（不调度）            |
| Init     | `random`                       |
| Model    | 100 个高斯，各向异性 + 透明度   |
| 参数分组  | 所有参数统一学习率              |

## 核心概念

### 高斯模型

每个 2D 高斯具有以下可学习参数：
- **中心** `(x, y)`：位置（sigmoid 约束到 [0,1]）
- **尺度** `(sigma_x, sigma_y)`：大小（softplus 约束为正）
- **旋转角** `theta`：方向（tanh 约束）
- **透明度** `alpha`：不透明度（sigmoid 约束到 [0,1]）
- **颜色** `(r, g, b)`：RGB 值（sigmoid 约束到 [0,1]）

### 渲染公式

各向异性高斯权重：

```
w_i(p) = alpha_i * exp(-0.5 * mahalanobis_i(p))
```

归一化加权混合：

```
I(p) = (sum_i w_i(p) * c_i + eps * bg_color) / (sum_i w_i(p) + eps)
```

### 参数分组学习率

不同参数（位置、尺度、颜色等）的梯度量级差异很大，统一学习率往往不是最优的。框架支持为每种参数设置独立的学习率倍率：

```python
config.optimizer.param_groups.center_lr_scale = 2.0   # 位置学更快
config.optimizer.param_groups.color_lr_scale = 0.5     # 颜色学更慢
```

有效学习率 = `base_lr * lr_scale`。

### 学习率调度器

训练过程中动态调整学习率，可以显著影响收敛行为：

```python
config.scheduler.name = "cosine"          # 余弦退火
config.scheduler.name = "warmup_cosine"   # 预热 + 余弦退火
config.scheduler.name = "step_decay"      # 阶梯衰减
config.scheduler.name = "constant"        # 不调度（基线）
```

调度器返回一个乘数 `scale ∈ [min_lr_scale, 1.0]`，每步作用于所有参数组。

## 你需要实现的内容

标有 `[学生文件]` 的文件包含 `raise NotImplementedError` 的 stub，你需要补全实现。

### 必做

| 文件 | 任务 |
|---|---|
| `optimizers/student_sgd.py` | 实现 SGD |
| `optimizers/student_momentum.py` | 实现 SGD + Momentum |
| `optimizers/student_adam.py` | 实现 Adam |
| `losses.py` | 实现至少 2 种新 loss（如 L1、MSE+L1） |
| `initializers/image_sample_init.py` 或 `bright_spot_init.py` | 实现至少 1 种图像感知初始化 |
| `schedulers.py` | 实现至少 1 种学习率调度器（如 cosine） |

### 选做（bonus）

| 文件 | 任务 |
|---|---|
| `optimizers/student_adamw.py` | 实现 AdamW（解耦权重衰减） |
| `optimizers/student_muon.py` | 实现 Muon（正交化优化器） |
| `optimizers/student_newton.py` | 实现 Newton 风格优化器 |

## 优化器进阶路线

```
SGD  →  SGD + Momentum  →  Adam  →  AdamW
 |         ↑                  ↑
 |    加入动量项           加入自适应学习率
 |    加速收敛             + 偏差校正
```

- **SGD**：`param -= lr * grad`
- **Momentum**：`v = mu * v + grad; param -= lr * v`
- **Adam**：自适应一阶/二阶矩估计 + 偏差校正
- **AdamW**：解耦权重衰减

## 如何切换配置

```python
from config import Config

config = Config()

# 切换优化器
config.optimizer.name = "student_adam"
config.optimizer.student_adam.lr = 1e-2

# 开启余弦调度
config.scheduler.name = "cosine"
config.scheduler.min_lr_scale = 0.01

# 参数分组
config.optimizer.param_groups.center_lr_scale = 2.0

# 切换初始化
config.initializer.name = "image_sample"

# 切换模型
config.model.use_anisotropic = True
config.model.use_alpha = True
```

## 运行模式

```bash
# 学生模式（默认）— 只有基线模块可用，其余需要自己实现
python main.py --mode student

# 教师模式 — 加载参考实现，所有模块可用
python main.py --mode teacher
```

## 竞赛自测

```bash
python experiments/run_competition.py --config experiments/my_competition_config.py
```

在 10 张测试图（5 张真实 RGB + 5 张合成高斯）上评测，输出每张图 PSNR 和平均 PSNR。

详细实验要求见 [`experiments/experiment1_spec.md`](experiments/experiment1_spec.md)。

## 评估指标

- **PSNR**（越高越好）：峰值信噪比，主要评分指标
- **MSE**（越低越好）：均方误差
- **MAE**（越低越好）：平均绝对误差

## 配置参考

所有参数在 `config.py` 中定义，按功能分组：

| 配置组 | 说明 |
|---|---|
| `config.system` | 种子、设备、输出目录 |
| `config.target` | 目标图像生成器、图像大小 |
| `config.model` | 高斯数量、各向异性/透明度开关 |
| `config.render` | 背景色、数值 epsilon |
| `config.train` | 训练步数、日志/保存频率 |
| `config.loss` | Loss 选择及超参数 |
| `config.optimizer` | 优化器选择、超参数、参数分组 |
| `config.scheduler` | 学习率调度器选择及超参数 |
| `config.initializer` | 初始化策略选择及超参数 |
| `config.visualization` | 视频导出设置 |
