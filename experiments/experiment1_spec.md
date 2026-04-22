# 优化实验 1：2D 高斯溅射图像拟合 — 模块消融与竞赛

## 实验背景

本实验基于 2D 高斯溅射（2D Gaussian Splatting）图像拟合任务。给定一张目标图像，系统用一组 2D 高斯函数的加权混合来逼近它，通过梯度优化不断改善拟合质量。

你拿到的基线代码已经可以运行：

```bash
python main.py
```

基线使用 MSE loss、PyTorch Adam 优化器、随机初始化、恒定学习率，可以产生一个初步的拟合结果。

**你的任务是**：实现缺失的模块，然后通过消融实验验证每个模块的作用，最后在竞赛中追求最高 PSNR。

---

## 基线代码已提供的模块

| 模块       | 已提供                          | 需要你实现                       |
| ---------- | ------------------------------- | -------------------------------- |
| Loss       | `mse`                           | 至少 2 种其他 loss                |
| Optimizer  | `torch_adam`（PyTorch 内置）     | `student_sgd`, `student_momentum`, `student_adam` |
| Scheduler  | `constant`（不调度）             | 至少 1 种调度器（如 cosine）      |
| Initializer| `random`                        | 至少 1 种图像感知初始化           |
| Model      | 各向异性/透明度开关已内置         | 无需实现，仅需对比分析            |
| 参数分组    | 接口已内置                       | 无需实现，仅需调参对比            |

---

## 第一部分：消融实验（60 分）

### 实验方法

消融实验的核心原则：**每次只改变一个模块，其余保持默认基线**，从而隔离该模块的贡献。

**默认基线配置：**

```python
config = Config()
config.model.num_gaussians = 200
config.train.num_steps = 500
config.system.seed = 42
config.target.image_size = 256
config.render.bg_color = (0.0, 0.0, 0.0)
config.target.name = "image"  # Starry Night
config.loss.name = "mse"
config.optimizer.name = "torch_adam"
config.optimizer.torch_adam.lr = 5e-2
config.scheduler.name = "constant"
config.initializer.name = "random"
config.model.use_anisotropic = True
config.model.use_alpha = True
```

你需要自己编写实验脚本来批量运行不同配置、收集结果、生成对比图表。这本身也是实验能力的一部分。

### 消融 A：Loss 函数（12 分）

**实现要求**：在 `losses.py` 中实现至少 2 种新的 loss 函数，并在 `build_loss` 中注册。

建议实现（也可自行设计其他 loss）：

| 编号 | Loss          | 说明                       |
| ---- | ------------- | -------------------------- |
| A1   | `mse`         | 均方误差（已提供，作为基线） |
| A2   | `l1`          | 绝对值误差                  |
| A3   | `mse_l1`      | MSE + L1 混合              |

**消融实验**：固定其他模块为基线，仅替换 loss，对比 PSNR / MSE / MAE。

**需要提交：**
- 你实现的 loss 函数代码
- 各组实验的数值结果表格（PSNR / MSE / MAE）
- loss 曲线对比图
- 文字分析（不超过 300 字）：哪个 loss 效果最好？为什么不同 loss 表现不同？

### 消融 B：初始化策略（12 分）

**实现要求**：在 `initializers/` 下实现至少 1 种图像感知初始化策略，并在 `initializers/factory.py` 中注册。

参考 `initializers/custom_initializer_template.py` 了解接口。你的初始化器会收到 `target_image`（`[H, W, 3]` 张量），可以利用它来决定高斯的初始位置和颜色。

建议实现思路（也可自行设计）：
- **图像采样初始化**：随机放置中心，但从目标图像对应位置采样颜色
- **亮度感知初始化**：将高斯中心放在目标图像亮度较高的区域

| 编号 | Initializer    | 说明                     |
| ---- | -------------- | ------------------------ |
| B1   | `random`       | 随机初始化（已提供，作为基线）|
| B2   | 你的实现        | 图像感知初始化             |

**消融实验**：固定其他模块为基线，仅替换初始化策略。

**需要提交：**
- 你实现的初始化策略代码
- 各组实验的 PSNR 数值表格
- loss 曲线对比图（重点观察前 50 步的收敛速度差异）
- 文字分析（不超过 300 字）：图像感知初始化的优势体现在哪里？

### 消融 C：优化器（12 分）

**实现要求**：补全以下三个优化器文件：

- `optimizers/student_sgd.py` — 最基础的梯度下降
- `optimizers/student_momentum.py` — SGD + 动量，收敛加速的关键一步
- `optimizers/student_adam.py` — 自适应学习率 + 动量

优化器的进阶路线：**SGD → SGD + Momentum → Adam**，每一步都引入一个核心改进。

参考 `optimizers/custom_optimizer_template.py` 了解接口。优化器需要实现 `zero_grad()` 和 `step()` 两个方法。注意每个参数组有独立的学习率 `group["lr"]`。

| 编号 | Optimizer         | 说明                                |
| ---- | ----------------- | ----------------------------------- |
| C1   | `torch_adam`      | PyTorch 内置 Adam（已提供，作为基线）|
| C2   | `student_sgd`     | 你实现的 SGD                        |
| C3   | `student_momentum`| 你实现的 SGD + Momentum             |
| C4   | `student_adam`    | 你实现的 Adam                       |

**消融实验**：固定其他模块为基线，仅替换优化器。

**正确性验证**：`student_adam` 和 `torch_adam` 在相同配置下的 PSNR 差异应小于 0.5 dB。如果差异过大，说明实现可能有误。

**需要提交：**
- 你实现的三个优化器代码
- 各组实验的 PSNR 数值表格
- loss 曲线对比图
- 文字分析（不超过 300 字）：SGD → Momentum → Adam 每一步带来了什么改进？Momentum 相比纯 SGD 提升了多少？你的 `student_adam` 与 `torch_adam` 是否一致？

### 消融 D：模型设计（12 分）

**无需实现代码**，仅需修改配置对比 4 种组合：

| 编号 | use_anisotropic | use_alpha | 说明                   |
| ---- | --------------- | --------- | ---------------------- |
| D1   | `False`         | `False`   | 各向同性 + 不透明       |
| D2   | `True`          | `False`   | 各向异性 + 不透明       |
| D3   | `False`         | `True`    | 各向同性 + 透明         |
| D4   | `True`          | `True`    | 各向异性 + 透明（基线）  |

**消融实验**：固定其他模块为基线，仅切换模型开关。

**需要提交：**
- 4 组实验的 PSNR 数值表格
- 最终重建结果的视觉对比图
- 文字分析（不超过 300 字）：各向异性和透明度各自带来多少 PSNR 提升？哪个影响更大？二者是否互补？

### 消融 E：学习率调度器（12 分）

**实现要求**：在 `schedulers.py` 中实现至少 1 种学习率调度器。

调度器是一个函数 `(step, total_steps) -> lr_multiplier`，返回值在 `[min_lr_scale, 1.0]` 之间，每步乘以所有参数组的 base_lr。

建议实现：

| 编号 | Scheduler       | 说明                                   |
| ---- | --------------- | -------------------------------------- |
| E1   | `constant`      | 不调度（已提供，作为基线）               |
| E2   | `cosine`        | 余弦退火                                |
| E3   | `warmup_cosine` | 线性预热 + 余弦退火（bonus）             |

**cosine 公式**：`scale = min_scale + 0.5 * (1 - min_scale) * (1 + cos(pi * step / total))`

**消融实验**：固定其他模块为基线，仅替换调度器。

**需要提交：**
- 你实现的调度器代码
- 各组实验的 PSNR 数值表格
- loss 曲线对比图（观察后期收敛行为的差异）
- 文字分析（不超过 300 字）：调度器对最终 PSNR 有多大影响？在什么训练阶段起关键作用？

---

## 第二部分：竞赛（40 分）

### 规则

自由组合所有模块的任意选项和超参数，目标是最大化 **10 张测试图的平均 PSNR**。

**硬约束（不可修改）：**

| 参数       | 值                     |
| ---------- | ---------------------- |
| 高斯数量    | 200                    |
| 背景色      | 黑色 `(0.0, 0.0, 0.0)` |
| 随机种子    | 42                     |
| 图像大小    | 256×256                |

**两个赛道：**

| 赛道            | 训练步数 | 侧重点         |
| --------------- | -------- | --------------- |
| Sprint（冲刺）   | 100 步   | 快速收敛能力     |
| Standard（标准） | 500 步   | 整体拟合质量     |

**可调节的内容：**
- Loss 函数及其超参数
- 初始化策略及其超参数
- 优化器及其超参数（学习率、动量等）
- 学习率调度器及其超参数
- 参数分组学习率倍率（`config.optimizer.param_groups`）
- 模型设计开关（各向异性、透明度）
- 可以自己实现新的优化器、初始化策略、loss 函数或调度器

**竞赛策略提示**：
- Sprint 赛道只有 100 步，初始化质量和学习率策略至关重要
- 参数分组可以让位置参数学得更快、颜色参数学得更稳
- 调度器可以在训练后期精细调整
- 不同的图像类型（真实图 vs 合成高斯）可能适合不同的策略

### 测试图像

竞赛使用 10 张测试图像（5 张真实 RGB + 5 张 txt 合成），评分取 10 张图的平均 PSNR：

**真实 RGB 图像（256×256）：**

| 编号 | 文件                                     | 描述                              |
| ---- | ---------------------------------------- | --------------------------------- |
| R1   | `data/Starry_Night_256.png`              | 梵高《星空》，复杂纹理和颜色渐变    |
| R2   | `data/competition/blackswan_256.png`     | 黑天鹅，自然场景，水面反射          |
| R3   | `data/competition/flamingo_256.png`      | 火烈鸟，细节纹理，水面倒影          |
| R4   | `data/competition/car-roundabout_256.png`| 汽车，城市场景，几何结构            |
| R5   | `data/competition/parkour_256.png`       | 跑酷人物，建筑背景，混合场景        |

**txt 合成高斯目标：**

| 编号 | 文件                                          | 描述                                   |
| ---- | --------------------------------------------- | -------------------------------------- |
| S1   | `data/competition/t3_sparse_colorful.txt`     | 15 个稀疏彩色高斯，简单场景             |
| S2   | `data/competition/t4_dense_cluster.txt`       | 30 个密集半透明高斯，测试遮挡处理       |
| S3   | `data/competition/t5_anisotropic_mix.txt`     | 20 个各向异性旋转高斯，测试方向拟合     |
| S4   | `data/examples/04_ten_translucent_stars.txt`  | 10 个半透明灰色高斯                    |
| S5   | `data/examples/05_ten_colorful_stars.txt`     | 10 个彩色高斯                          |

### 评分标准

每个赛道独立排名，按 **10 张图平均 PSNR** 从高到低排序：

| 排名百分位 | 得分（每赛道满分 20 分） |
| ---------- | ----------------------- |
| Top 10%    | 20                      |
| Top 30%    | 17                      |
| Top 50%    | 14                      |
| Top 70%    | 11                      |
| 其余       | 8（参与分）              |

两个赛道总分 = Sprint 得分 + Standard 得分，满分 40 分。

### 如何自测

```bash
python experiments/run_competition.py --config experiments/my_competition_config.py
```

脚本会依次在 10 张测试图上运行你的配置，输出每张图的 PSNR 和平均 PSNR。

### 提交要求

1. 提交 `my_competition_config.py`，包含两个函数（见下方模板）
2. 如有自定义模块（优化器 / 初始化 / loss / 调度器），一并提交源文件
3. 简短说明（不超过 500 字）：你的配置策略和选择理由

### 提交模板

```python
from config import Config


def get_sprint_config() -> Config:
    """Sprint 赛道：100 步，追求快速收敛。"""
    config = Config()
    # === 硬约束，不要修改 ===
    config.model.num_gaussians = 200
    config.render.bg_color = (0.0, 0.0, 0.0)
    config.system.seed = 42
    config.target.image_size = 256
    config.train.num_steps = 100
    # === 以下自由配置 ===
    config.loss.name = "mse"
    config.initializer.name = "random"
    config.optimizer.name = "torch_adam"
    config.optimizer.torch_adam.lr = 5e-2
    config.scheduler.name = "constant"
    config.optimizer.param_groups.center_lr_scale = 1.0
    config.optimizer.param_groups.color_lr_scale = 1.0
    config.model.use_anisotropic = True
    config.model.use_alpha = True
    return config


def get_standard_config() -> Config:
    """Standard 赛道：500 步，追求最高 PSNR。"""
    config = Config()
    # === 硬约束，不要修改 ===
    config.model.num_gaussians = 200
    config.render.bg_color = (0.0, 0.0, 0.0)
    config.system.seed = 42
    config.target.image_size = 256
    config.train.num_steps = 500
    # === 以下自由配置 ===
    config.loss.name = "mse"
    config.initializer.name = "random"
    config.optimizer.name = "torch_adam"
    config.optimizer.torch_adam.lr = 5e-2
    config.scheduler.name = "constant"
    config.optimizer.param_groups.center_lr_scale = 1.0
    config.optimizer.param_groups.color_lr_scale = 1.0
    config.model.use_anisotropic = True
    config.model.use_alpha = True
    return config
```

---

## 总评分

| 部分                  | 满分 |
| --------------------- | ---- |
| 消融 A：Loss 函数      | 12   |
| 消融 B：初始化策略      | 12   |
| 消融 C：优化器         | 12   |
| 消融 D：模型设计       | 12   |
| 消融 E：学习率调度器    | 12   |
| 竞赛 Sprint 赛道       | 20   |
| 竞赛 Standard 赛道     | 20   |
| **合计**              | **100** |
