# 本地环境与流程验证记录

日期：2026-10-20
机器：Windows + NVIDIA RTX 3060 12 GB

## 1. Conda 环境

- 环境名：`sar_method`
- 路径：`D:\Anaconda3\envs\sar_method`
- Python：3.12.15
- PyTorch：2.7.1+cu118，`torch.cuda.is_available() = True`
- NumPy：2.2.6
- SciPy：1.13.1
- OpenCV：4.11.0
- Pillow：11.3.0
- pandas：2.2.2
- matplotlib：3.9.2
- contourpy：1.3.3

`pip check`：`No broken requirements found.`

检查命令：

```powershell
conda run --no-capture-output -n sar_method python run.py --check-env
```

结果：通过。

## 2. 代码契约检查

```powershell
cd registration
conda run --no-capture-output -n sar_method python verify_environment.py
```

结果：`PASS`，8 项检查全部通过（仿射方向、SimCLR 前后向/断点、迭代反馈、四指标导出等）。CPU 合成检查，未用真实数据。

## 3. 一键完整流程（复原 + 配准）

```powershell
conda run --no-capture-output -n sar_method python run.py `
  --input-dir data\example\demo `
  --output-dir outputs\conda_smoke `
  --restoration-steps 2 --contrastive-epochs 1
```

- 退出码：0
- 状态：`accepted`，`effective=true`
- 四个指标：`Nred=19`，`RMSEall=0.6864`，`RMSEloo=0.8241`，`Pquad=0.4211`
- 复原日志确认损失配方：`name=selected_final`、`gradient=0.02236068`、`structure=0.70710678`；第二轮 `teacher_effective=1.7320508`
- 配准实际读取的是 `restored_sensed.png`（raw），不是 `_radiometric.png`

注意：这次只跑了 2 次复原迭代和 1 个 SimCLR epoch，仅用于验证流程能跑通，指标没有科研意义。正常使用默认 `300` 次复原迭代、`100` 个 epoch。

## 4. 跳过复原、只跑配准

```powershell
conda run --no-capture-output -n sar_method python run.py `
  --sensed data\example\demo\demo_passive_PGA64.jpg `
  --reference data\example\demo\demo_active.jpg `
  --round1 <round1.png> --round2 <round2.png> `
  --output-dir outputs\reg_wrapper_smoke --contrastive-epochs 1
```

- 流程内部退出码：0
- 状态：`learned_update_accepted`，`effective=true`
- 四个指标：`Nred=20`，`RMSEall=0.6648`，`RMSEloo=0.7188`，`Pquad=0.5000`

## 5. 结论

`D:\Autodl\method` 已可在 `sar_method` 环境中直接运行；单图对、多图对、只跑配准三种入口都已验证。
