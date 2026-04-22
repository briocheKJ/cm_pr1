# 数据说明

## 目录结构

| 路径 | 内容 |
| ---- | ---- |
| `Starry_Night_256.png` | 默认真实目标图 |
| `competition/` | 竞赛测试图像与 txt 目标 |
| `examples/` | 小型 txt 示例，便于调试 |

## 数据类型

本项目包含两类目标数据。

### 1. 真实 RGB 图像

直接读取图片，裁切 / 缩放到配置中的尺寸。

主要文件：

| 文件 | 用途 |
| ---- | ---- |
| `Starry_Night_256.png` | 默认训练目标 |
| `competition/*.png` | 竞赛测试图像 |

### 2. txt 高斯目标

通过 txt 文件描述若干 Gaussian，再由目标生成器渲染成图像。

支持的常见格式：

| 格式 | 含义 |
| ---- | ---- |
| `x y sigma r g b` | 各向同性，无 alpha |
| `x y sigma alpha r g b` | 各向同性，有 alpha |
| `x y sigma_x sigma_y theta r g b` | 各向异性，无 alpha |
| `x y sigma_x sigma_y theta alpha r g b` | 各向异性，有 alpha |

### 3. 示例数据

`examples/` 里的文件用于调试和展示：

| 文件 | 用途 |
| ---- | ---- |
| `01_single_gray_isotropic_star.txt` | 单个各向同性高斯 |
| `02_single_gray_anisotropic_star.txt` | 单个各向异性高斯 |
| `03_ten_gray_stars.txt` | 多个灰色高斯 |
| `04_ten_translucent_stars.txt` | 多个半透明高斯 |
| `05_ten_colorful_stars.txt` | 多个彩色高斯 |

## 使用建议

| 场景 | 建议使用的数据 |
| ---- | ---- |
| 跑通训练闭环 | `Starry_Night_256.png` |
| 调试目标生成器 | `examples/*.txt` |
| 做竞赛评测 | `competition/` |
