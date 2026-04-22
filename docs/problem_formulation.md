# 优化问题形式化

## 任务定义

给定一张目标 RGB 图像 `I_target`，用 `N` 个 2D Gaussian 的叠加结果去拟合它。

每个 Gaussian 至少包含：

- 位置 `mu_i = (x_i, y_i)`
- 尺度 `sigma_i`
- 颜色 `c_i = (r_i, g_i, b_i)`

若启用扩展形式，还可以包含：

- 各向异性尺度 `(sigma_x, sigma_y)`
- 旋转角 `theta`
- 透明度 `alpha`

## 参数化

代码里优化的是原始参数，再通过约束映射得到有效参数：

- 位置：`center = sigmoid(center_raw)`
- 尺度：`scale = softplus(scale_raw) + 1e-4`
- 颜色：`color = sigmoid(color_raw)`
- 角度：`theta = pi * tanh(angle_raw)`（若启用各向异性）
- 透明度：`alpha = sigmoid(alpha_raw)`（若启用 alpha）

## 渲染

对像素位置 `p = (u, v)`，第 `i` 个 Gaussian 的基础权重写成：

```text
g_i(p) = exp( - d_i(p)^2 / 2 )
```

其中：

- 各向同性时，`d_i(p)^2 = ||p - mu_i||^2 / sigma_i^2`
- 各向异性时，`d_i(p)^2` 为旋转椭圆坐标下的 Mahalanobis 距离

若启用透明度，则：

```text
w_i(p) = alpha_i * g_i(p)
```

否则：

```text
w_i(p) = g_i(p)
```

最终颜色由归一化加权平均得到：

```text
I(p) = (sum_i w_i(p) * c_i + eps * bg) / (sum_i w_i(p) + eps)
```

其中 `bg = (0, 0, 0)`，`eps = 1e-6`。

## 优化目标

训练的目标是最小化重建图像与目标图像之间的差异：

```text
min_theta  L( I_render(theta), I_target )
```

其中 `theta` 表示全部高斯参数，`L` 可以是：

- `mse`
- `l1`
- `charbonnier`
- `mse_l1`
- `mse_edge`

## 评估指标

训练结束后主要比较：

- `MSE`：越低越好
- `MAE`：越低越好
- `PSNR`：越高越好

## 本实验关注什么

本实验不仅比较“最终能否拟合好”，也比较：

- 收敛速度
- 收敛稳定性
- 对初始化的敏感性
- 不同模型开关与优化策略的影响
