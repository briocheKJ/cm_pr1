# 竞赛说明

## 目标

自由组合模块和超参数，在测试集上获得尽可能高的平均 PSNR。

建议先完成必做模块和消融实验，再进入竞赛部分。竞赛部分将根据你在实验报告中报告的PSNR进行全班排名，以确定你的最终分数。

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
| 图像大小 | `256 x 256` |

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

竞赛使用 10 张测试图像：

| 编号 | 类型 | 文件 |
| ---- | ---- | ---- |
| R1 | RGB | `data/real_images/Starry_Night_256.png` |
| R2 | RGB | `data/real_images/blackswan_256.png` |
| R3 | RGB | `data/real_images/flamingo_256.png` |
| R4 | RGB | `data/real_images/car-roundabout_256.png` |
| R5 | RGB | `data/real_images/parkour_256.png` |
| S1 | txt | `data/txt/t1_translucent_stars.txt` |
| S2 | txt | `data/txt/t2_colorful_stars.txt` |
| S3 | txt | `data/txt/t3_sparse_colorful.txt` |
| S4 | txt | `data/txt/t4_dense_cluster.txt` |
| S5 | txt | `data/txt/t5_anisotropic_mix.txt` |

最终成绩取 10 张图的平均 PSNR。

真实图像示意：

<p align="center">
  <img src="competition_real_images.png" width="720" alt="Competition real images">
</p>

合成目标示意：

<p align="center">
  <img src="competition_synthetic_images.png" width="720" alt="Competition synthetic images">
</p>

## 评分

每个赛道单独排名：

| 排名百分位 | 得分 |
| ---- | ---- |
| Top 10% | 20 |
| Top 30% | 17 |
| Top 50% | 14 |
| Top 70% | 11 |
| 其余 | 8 |

总分 = Sprint + Standard，满分 40。

