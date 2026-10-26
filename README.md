# SAR 复原 + 配准一键流程

本目录是服务器配准/复原代码的本地整理版，默认使用用户于 2026-10-20 指定的损失配方：

```text
Charbonnier 像素损失
+ 0.2 × 梯度损失
+ 0.3 × LNCC/结构损失
+ 1.0 × teacher 一致性（仅第二轮有上一轮 checkpoint 时）
```

配方同时写在 `configs/loss_selected.json` 和 `restoration/loss_candidates.py` 的 `selected_final` 中。

SimCLR 当前默认对比损失为 `0.5 × NT-Xent(z) + 1.0 × NT-Xent(h)`，温度为 `0.15`，对应 `LearningConfig.z_weight=0.5`、`h_weight=1.0`；这两个系数独立设置，不要求和为 1。

## 1. 环境

已经创建好 Conda 环境 `sar_method`（Python 3.12 + PyTorch 2.7.1 cu118 + OpenCV 4.11）。

新机器上重建：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

检查：

```powershell
conda run --no-capture-output -n sar_method python run.py --check-env
```

不需要先 `conda activate`，用 `conda run --no-capture-output -n sar_method python ...` 即可；也可以 `conda activate sar_method` 后再运行。

## 2. 一条命令跑完整流程

默认示例数据（`data/example/demo/demo_passive_PGA64.jpg` + `demo_active.jpg`）：

```powershell
conda run --no-capture-output -n sar_method python run.py --input-dir data/example/demo --output-dir outputs/example
```

单图对：

```powershell
conda run --no-capture-output -n sar_method python run.py `
  --sensed D:\data\scene_passive_PGA64.jpg `
  --reference D:\data\scene_active.jpg `
  --output-dir D:\runs\scene01
```

流程会先做两轮 NAFNet 复原，再把 **raw `restored_sensed.png`**（不是 `_radiometric.png` 亮度增强版）送入三层 SAR-SIFT/SimCLR 配准，最后输出五个指标。

复原阶段内部的下一轮 SAR-SIFT、目标 patch 更新和实验评估也统一使用 raw 复原图，不再生成或使用亮度增强图。两轮网络输入始终为原始 sensed 图。SimCLR 更新的变换只反馈到后续匹配迭代，不会回到 NAFNet 重做复原。

旧运行的内部反馈图像策略不同，不能直接续用；修改后请使用新的输出目录重新运行。

## 3. 换数据

### 单图对

```powershell
python run.py --sensed <passive/sensed 图> --reference <active/reference 图> --output-dir <输出目录>
```

两幅图必须同尺寸；`--sensed` 是待配准图，`--reference` 是参考图。

### 一个目录多图对

```powershell
python run.py --input-dir D:\data\my_scenes --expected-pairs 3 --output-dir D:\runs\my_scenes
```

目录内文件名按下面规则成对：

```text
<key>_active.jpg
<key>_passive.jpg
或
<key>_active.jpg
<key>_passive_PGA64.jpg
```

支持 `.jpg/.jpeg/.png/.tif/.tiff`。每个 key 必须正好有一张 active 和一张 passive。

`--input-dir` 只扫描该目录的直接子文件，不递归扫描子目录。active 是参考图，passive 是待配准图。目录中的所有受支持图像文件都必须符合上述配对规则。

Windows 下无需先切换目录或激活环境，可直接批量运行：

```powershell
conda run --no-capture-output -n sar_method python D:\Autodl\method\run.py --input-dir "D:\data\my_scenes" --output-dir "D:\Autodl\method\outputs\batch_zh05_1" --workers 1
```

把输入目录替换成实际路径，输出使用新的目录。所有图对完成后，`batch_metrics.csv` 汇总五项指标，各图对的图像、变换和详细指标保存在各自子目录中。

### 已经复原好，只跑配准

```powershell
python run.py `
  --sensed D:\data\scene_passive.jpg `
  --reference D:\data\scene_active.jpg `
  --round1 D:\restored\round1.png `
  --round2 D:\restored\round2.png `
  --output-dir D:\runs\reg_only
```

这条路径跳过复原训练，直接跑配准。

## 4. 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | `cuda` | `cuda`、`cuda:0` 或 `cpu` |
| `--restoration-steps` | `300` | 每轮 NAFNet 迭代数 |
| `--restoration-seed` | `2026` | 复原随机种子 |
| `--registration-seed` | `20260709` | 配准随机种子 |
| `--contrastive-epochs` | `100` | SimCLR 轮数 |
| `--workers` | `1` | 多图对并行进程数，1–3 |
| `--gradient-weight` | `0.2` | 覆盖选中梯度权重 |
| `--structure-weight` | `0.3` | 覆盖选中 LNCC 权重 |
| `--teacher-weight` | `1.0` | 覆盖选中 teacher 权重 |
| `--dry-run` | 关闭 | 只打印命令，不执行 |

快速冒烟（质量会明显下降，只用于确认能跑通）：

```powershell
python run.py --input-dir data/example/demo --output-dir outputs/smoke `
  --restoration-steps 20 --contrastive-epochs 5
```

## 5. 输出

```text
outputs/example/
  batch_metrics.csv            # 每个图对的 Nred / RMSEall / RMSEloo / Pquad / NCC
  batch_summary.json
  input_pairs.json
  run_config.json              # 本次参数和损失权重
  <pair_key>/
    restoration/
      restoration_after_round1/restored_sensed.png
      restoration_after_round2/restored_sensed.png
    registration/
      registration_results/<pair_key>/
        metrics.json
        registration_metrics.txt
        registered_sensed.png
        registration_overlay.png
        final_registration_matches.png
        validation_report.json
    results/                   # 上面 registration_results 的副本
```

最终输出五个指标（默认保留全部通过几何筛选的内点，不再限制为 24 个；也不因内点超过 30 个而拒绝结果）：

- `Nred`：最终保留的几何内点数；
- `RMSEall`：保留内点的重投影 RMSE（像素）；
- `RMSEloo`：留一仿射预测 RMSE（像素）；
- `Pquad`：四个象限中最大内点占比，越低空间分布越均匀。
- `NCC`：原始 sensed 图经最终变换后，与参考图在固定目标验证窗口内的 NCC 均值，越高越好。采用 log1p 和 sigma=2 高斯平滑，与 `SimCLR_loss` 实验口径一致；验证窗口与图像引导拟合窗口分开，但不是训练外测试集。

`NCC` 写入最终和逐轮 `metrics.json`、`registration_metrics.txt`、逐轮与批量 CSV、单对摘要和终端输出。未执行配准或缺少验证窗口时不提供有效 NCC。直接调用配准入口时，`--max-final-matches 0` 表示不限，正数仍可显式指定上限。

三层 SAR-SIFT 的 NCC 同样写入 `fusion_report.json` 和结果目录中的 `four_stage_metrics.csv/json`。为公平比较几何变换，Original、round1、round2 和最终结果均使用**原始 passive 图**经各自变换后的图像，在同一参考图、同一组固定验证窗口上计算 NCC；不把复原后的灰度变化混入这个对比。SAR-SIFT 没有有效变换时，其 NCC 留空。

已有完整输出可以补算前三层 NCC，无需重新训练：

```powershell
conda run --no-capture-output -n sar_method python D:\Autodl\method\registration\update_saved_stage_ncc.py --batch-root D:\Autodl\method\outputs\Sentinel_1
```

脚本保留更新前的报告备份，补算每对的四阶段表，并在批量根目录生成 `batch_four_stage_metrics.csv` 和 `registration_failure_audit.json`。若最终没有接受任何 SimCLR 更新，`Nred=0`、几何误差为空，配准图和 NCC 对应保留的共识 T0，并不代表成功的精配准。

注意：这些指标建立在“最终保留内点”上，不同方法/数据的 `Nred` 可能不同，RMSE 不是同一批点上的绝对误差，也不是有地面真值的精度。

## 6. 目录

```text
run.py                     # 一键入口
run_full_pipeline.py       # 复原 + 配准主流程
registration/              # SAR-SIFT / SimCLR 配准
restoration/               # NAFNet 复原、损失、目标筛选
models/simclr_initial.pt   # 配准 warm start
configs/loss_selected.json # 最终损失权重
data/example/demo/         # 示例图对
outputs/                   # 运行输出
```

## 7. 常见问题

- **输出目录已存在**：完整流程会按 `--continue-existing` 续跑；已经完成的图对会复用，未完成的会保留旧目录后重跑。要全新重跑请换一个 `--output-dir`。
- **显存不足**：把 `--workers` 设为 1，或 `--device cpu`（会慢很多）。
- **两幅图尺寸不一致**：复原训练要求同尺寸，请先裁到同一网格。
- **中断后继续**：用同一个 `--output-dir` 再执行一次即可。
- **`KMP_DUPLICATE_LIB_OK`**：`run.py` 已自动设置，避免 Anaconda 的 OpenMP 冲突。


### 2026-10-21：最终配准只要求可估计变换

取消最终匹配/验收的硬质量门槛：描述子相似度和双向差值、搜索半径截断、同目标限制、邻域位移一致性、最少6个内点、空间分散/覆盖/舰船岸线配额、支持点误差、留一误差、相对前一变换的误差改善、最终RMSE/Pquad比较，以及仿射翻转/尺度/长宽比限制。候选融合不再按最低置信度剔除。NCC及其余四项指标只记录，不影响接收。

双向最近邻仍用于建立对应，距离仅作软评分；不足3对时尝试全局一对一匹配。RANSAC仍作为稳健估计器使用，其内点不设数量质量门槛；不能估计时尝试全部候选的加权最小二乘。仿射估计至少需要3对非共线对应，变换必须有限且非奇异。`--min-final-matches`、`--max-final-matches`、相似度阈值等旧参数保留兼容，但不再决定最终接收或筛点上限。

每次能估计的变换直接反馈到下一轮参考图裁块；后续无法估计新变换时保留最近一次变换；没有可估计的学习变换时保留SAR-SIFT初始变换，明确记录`transform_source=SAR_SIFT_T0`，不虚报学习匹配点。参考/待配准点不足以运行学习阶段或不能构建NCC窗口时，也不因此丢弃已有初始变换。完整128像素裁块的边界约束及输入文件、缓存身份检查保留。

五组已训练模型的独立检查结果见 `outputs/Sentinel_1_unrestricted_check/summary.json`，没有覆盖原实验或重训网络。解除质量门槛会接受质量较差的变换，输出成功不能代表配准精度提高。
