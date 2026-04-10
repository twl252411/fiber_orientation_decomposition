# Fiber Orientation Decomposition / 纤维取向分解

## 1. Overview / 项目概述

**English**  
This repository provides a practical workflow for short-fiber composite analysis around orientation states (`a1-a3`, `b1-b6`). It covers:
- stochastic fiber placement and orientation generation,
- Digimat MF/FE input generation and batch execution,
- ENG result parsing to clean tensor text files,
- interpolation-based orientation decomposition for stiffness, CTE, and thermal conductivity.

The project is designed for engineering studies where baseline orientation states are solved first, then interpolated to a target orientation tensor.

**中文**  
本仓库提供了一个面向短纤维复合材料的实用流程，围绕取向状态（`a1-a3`、`b1-b6`）进行分析，包含：
- 随机纤维位置与角度生成，
- Digimat MF/FE 输入文件批量生成与求解，
- ENG 结果自动提取为标准张量文本，
- 基于插值的取向分解（刚度、CTE、导热率）。

该项目适用于先计算基准取向状态，再对目标取向张量进行插值预测的工程场景。

---

## 2. Repository Layout / 仓库结构

**English**

```text
fiber_orientation_decomposition/
├─ point_angle_files/              # Generated point/angle inputs (a1-a3, b1-b6)
├─ rve_generator/                  # Random packing + orientation optimization
│  ├─ main.py
│  └─ distance3d/
├─ rve_abaqus/                     # Abaqus CAE script to build fiber/matrix RVE
│  └─ rve_fibers.py
├─ digimatMF_analysis/             # Digimat MF pipeline
│  ├─ mat_generator.py             # Generate Analysis_<index>_<type>.mat
│  ├─ digimatMF_runner.py          # Run Digimat MF batch
│  ├─ eng_results_mf.py            # Parse ENG -> Stiffness/CTE/Conductivity txt
│  ├─ Template_tm.mat
│  └─ Template_etc.mat
├─ digimatFE_analysis/             # Digimat FE pipeline
│  ├─ daf_generator_fe.py          # Generate Analysis_<index>_<type>.daf
│  ├─ digimatFE_runner.py          # Run Digimat FE batch
│  └─ eng_results_fe.py            # Parse ENG -> Stiffness/CTE/Conductivity txt
├─ orien_decomp_method/            # Orientation decomposition/interpolation
│  ├─ main.py
│  ├─ eigen_utils.py
│  ├─ quadratic_interpolation.py
│  └─ tensor_utils.py
└─ orien_decomp_results/           # Decomposition outputs
```

**中文**

```text
fiber_orientation_decomposition/
├─ point_angle_files/              # 位置/角度输入（a1-a3, b1-b6）
├─ rve_generator/                  # 随机填充与取向优化
├─ rve_abaqus/                     # Abaqus 脚本：构建纤维/基体 RVE
├─ digimatMF_analysis/             # Digimat MF 流程（MAT 生成、求解、结果提取）
├─ digimatFE_analysis/             # Digimat FE 流程（DAF 生成、求解、结果提取）
├─ orien_decomp_method/            # 取向分解与插值
└─ orien_decomp_results/           # 分解结果输出
```

---

## 3. Requirements / 环境要求

### 3.1 Python / Python 依赖

**English**
- Python `3.10+` recommended.
- Main packages used by the repository:
  - `numpy`
  - `scipy`
  - `numba`
  - `pytransform3d`

Install:

```powershell
pip install numpy scipy numba pytransform3d
```

**中文**
- 建议使用 Python `3.10+`。
- 仓库主要依赖：
  - `numpy`
  - `scipy`
  - `numba`
  - `pytransform3d`

安装命令：

```powershell
pip install numpy scipy numba pytransform3d
```

### 3.2 Commercial Tools / 商业软件依赖

**English**
- Digimat MF and/or Digimat FE are required to run solver workflows.
- `rve_abaqus/rve_fibers.py` requires Abaqus Python/CAE environment.
- Script defaults currently point to:
  - `C:\MSC.Software\Digimat\2023.1\DigimatMF\exec\...`
  - `C:\MSC.Software\Digimat\2023.1\DigimatFE\exec\DigimatFE.bat`

Please update paths in runner scripts if your installation differs.

**中文**
- 求解流程需要 Digimat MF 和/或 Digimat FE。
- `rve_abaqus/rve_fibers.py` 需要在 Abaqus Python/CAE 环境下运行。
- 当前脚本默认路径为：
  - `C:\MSC.Software\Digimat\2023.1\DigimatMF\exec\...`
  - `C:\MSC.Software\Digimat\2023.1\DigimatFE\exec\DigimatFE.bat`

若本机安装路径不同，请在对应 runner 脚本中修改。

---

## 4. Quick Start (Recommended Order) / 快速开始（推荐顺序）

**English**
1. Generate or prepare `points_*.txt` + `angles_*.txt` (and periodic files).
2. Run Digimat MF or FE batch simulations for selected states.
3. Extract parsed tensors (`Stiffness`, `CTE`, `Conductivity`) from ENG files.
4. Run orientation decomposition interpolation (`orien_decomp_method/main.py`).

**中文**
1. 生成或准备 `points_*.txt` 与 `angles_*.txt`（以及周期扩展文件）。
2. 对选定状态运行 Digimat MF 或 FE 批量计算。
3. 从 ENG 文件提取标准张量结果（`Stiffness`、`CTE`、`Conductivity`）。
4. 运行取向分解插值（`orien_decomp_method/main.py`）。

---

## 5. Workflow A: RVE Point/Angle Generation / 流程A：RVE 点位与角度生成

### 5.1 Generate points and angles / 生成点位与角度

**English**
From `rve_generator/` run:

```powershell
cd rve_generator
python main.py
```

Key config in `rve_generator/main.py`:
- `ORI_ID`: orientation case tag (e.g., `a1`, `b6`)
- `RANDOM_SEED`: reproducibility (`None` for random)
- `MAX_ITERATIONS`, `LOG_INTERVAL`
- `OUTPUT_TAG`: output suffix override

Outputs are written to `point_angle_files/`:
- `points_<tag>.txt`
- `angles_<tag>.txt` (degree)
- `peri_points_<tag>.txt`
- `peri_angles_<tag>.txt` (degree)

**中文**
在 `rve_generator/` 下运行：

```powershell
cd rve_generator
python main.py
```

`rve_generator/main.py` 关键参数：
- `ORI_ID`：取向状态标签（如 `a1`、`b6`）
- `RANDOM_SEED`：随机种子（`None` 表示每次随机）
- `MAX_ITERATIONS`、`LOG_INTERVAL`
- `OUTPUT_TAG`：输出标签覆盖

输出到 `point_angle_files/`：
- `points_<tag>.txt`
- `angles_<tag>.txt`（角度制）
- `peri_points_<tag>.txt`
- `peri_angles_<tag>.txt`（角度制）

### 5.2 Build CAE with Abaqus (optional) / Abaqus 构建 CAE（可选）

**English**
`rve_abaqus/rve_fibers.py` reads `peri_points_*.txt` + `peri_angles_*.txt` and builds CAE models (`fiber_rve_<tag>.cae`). Run it using Abaqus Python/CAE command line.

**中文**
`rve_abaqus/rve_fibers.py` 会读取 `peri_points_*.txt` 与 `peri_angles_*.txt` 并生成 `fiber_rve_<tag>.cae`，需通过 Abaqus 命令行运行。

---

## 6. Workflow B: Digimat MF Pipeline / 流程B：Digimat MF 流程

### 6.1 Generate MAT input files / 生成 MAT 输入

**English**

```powershell
cd digimatMF_analysis
python mat_generator.py
```

This creates files like:
- `Analysis_a1_tm.mat`
- `Analysis_a1_etc.mat`
- ... for configured `SELECTED_INDEXES` and `SELECTED_ANALYSIS_TYPES`.

**中文**

```powershell
cd digimatMF_analysis
python mat_generator.py
```

将根据 `SELECTED_INDEXES` 与 `SELECTED_ANALYSIS_TYPES` 生成：
- `Analysis_a1_tm.mat`
- `Analysis_a1_etc.mat`
- 等等。

### 6.2 Run Digimat MF batch / 批量运行 Digimat MF

**English**

```powershell
cd digimatMF_analysis
python digimatMF_runner.py
```

By default, results are run in per-case temp folders such as:
- `tmp_a1_tm/`
- `tmp_b6_etc/`

**中文**

```powershell
cd digimatMF_analysis
python digimatMF_runner.py
```

默认会在每个工况的临时目录中求解，例如：
- `tmp_a1_tm/`
- `tmp_b6_etc/`

### 6.3 Parse ENG outputs / 提取 ENG 结果

**English**

```powershell
cd digimatMF_analysis
python eng_results_mf.py
```

Generated parsed files in `digimatMF_analysis/` include:
- `Analysis_<index>_Stiffness.txt` (6x6, reordered)
- `Analysis_<index>_CTE.txt` (3x3)
- `Analysis_<index>_Conductivity.txt` (3x3)

**中文**

```powershell
cd digimatMF_analysis
python eng_results_mf.py
```

将在 `digimatMF_analysis/` 目录下生成：
- `Analysis_<index>_Stiffness.txt`（6x6，已重排）
- `Analysis_<index>_CTE.txt`（3x3）
- `Analysis_<index>_Conductivity.txt`（3x3）

---

## 7. Workflow C: Digimat FE Pipeline / 流程C：Digimat FE 流程

### 7.1 Generate DAF input files / 生成 DAF 输入

**English**

```powershell
cd digimatFE_analysis
python daf_generator_fe.py
```

This reads `point_angle_files/points_<index>.txt` and `angles_<index>.txt` and writes:
- `Analysis_<index>_tm.daf`
- `Analysis_<index>_etc.daf`

**中文**

```powershell
cd digimatFE_analysis
python daf_generator_fe.py
```

该步骤读取 `point_angle_files/points_<index>.txt` 与 `angles_<index>.txt`，生成：
- `Analysis_<index>_tm.daf`
- `Analysis_<index>_etc.daf`

### 7.2 Run Digimat FE batch / 批量运行 Digimat FE

**English**

```powershell
cd digimatFE_analysis
python digimatFE_runner.py
```

Temporary folders are created per case (`tmp_<index>_<type>`), where `.eng` results are produced.

**中文**

```powershell
cd digimatFE_analysis
python digimatFE_runner.py
```

每个工况会生成临时目录（`tmp_<index>_<type>`），并在其中得到 `.eng` 文件。

### 7.3 Parse ENG outputs / 提取 ENG 结果

**English**

```powershell
cd digimatFE_analysis
python eng_results_fe.py
```

Parsed outputs are written as:
- `Analysis_<index>_Stiffness.txt`
- `Analysis_<index>_CTE.txt`
- `Analysis_<index>_Conductivity.txt`

**中文**

```powershell
cd digimatFE_analysis
python eng_results_fe.py
```

提取后输出为：
- `Analysis_<index>_Stiffness.txt`
- `Analysis_<index>_CTE.txt`
- `Analysis_<index>_Conductivity.txt`

---

## 8. Workflow D: Orientation Decomposition / 流程D：取向分解插值

**English**
From `orien_decomp_method/`:

```powershell
cd orien_decomp_method
python main.py
```

Configure at top of `main.py`:
- `SOURCE = "fe"` or `"mf"`
- `INTERP = "linear"` or `"quadratic"`
- `ANALYSIS_TYPES = ["elastic", "cte", "etc"]` (any subset)
- `ORIENTATION_CASE = 1 | 2 | 3`

Output files are written to `orien_decomp_results/`, for example:
- `Analysis_linear_fe_Stiffness_V.txt`
- `Analysis_linear_fe_Stiffness_R.txt`
- `Analysis_linear_fe_CTE_V.txt`
- `Analysis_linear_fe_ETC_V.txt`

Where:
- `V` = Voigt-type averaging result
- `R` = Reuss-type averaging result

**中文**
在 `orien_decomp_method/` 下运行：

```powershell
cd orien_decomp_method
python main.py
```

在 `main.py` 顶部配置：
- `SOURCE = "fe"` 或 `"mf"`
- `INTERP = "linear"` 或 `"quadratic"`
- `ANALYSIS_TYPES = ["elastic", "cte", "etc"]`（可选子集）
- `ORIENTATION_CASE = 1 | 2 | 3`

输出写入 `orien_decomp_results/`，例如：
- `Analysis_linear_fe_Stiffness_V.txt`
- `Analysis_linear_fe_Stiffness_R.txt`
- `Analysis_linear_fe_CTE_V.txt`
- `Analysis_linear_fe_ETC_V.txt`

其中：
- `V` 表示 Voigt 型平均
- `R` 表示 Reuss 型平均

---

## 9. Data and Naming Convention / 数据与命名约定

**English**
- Orientation state tags:
  - `a1-a3`: predefined anisotropic tensors
  - `b1-b6`: baseline/reference tensor family
- Analysis types:
  - `tm`: thermo-mechanical (stiffness + CTE)
  - `etc`: effective thermal conductivity
- Typical file names:
  - `Analysis_<index>_<type>.mat` / `.daf`
  - `Analysis_<index>_<type>.eng`
  - `Analysis_<index>_Stiffness.txt`, `..._CTE.txt`, `..._Conductivity.txt`

**中文**
- 取向状态标签：
  - `a1-a3`：预定义各向异性取向张量
  - `b1-b6`：基准/参考取向族
- 分析类型：
  - `tm`：热-力学（刚度 + CTE）
  - `etc`：有效导热率
- 常见文件命名：
  - `Analysis_<index>_<type>.mat` / `.daf`
  - `Analysis_<index>_<type>.eng`
  - `Analysis_<index>_Stiffness.txt`、`..._CTE.txt`、`..._Conductivity.txt`

---

## 10. Common Issues / 常见问题

**English**
1. `ModuleNotFoundError` when running scripts:  
   Run from the script directory (for example `cd rve_generator` before `python main.py`), because some imports are local-module style.

2. Digimat executable not found:  
   Update hard-coded paths in `digimatMF_runner.py` / `digimatFE_runner.py` to your local installation.

3. ENG parsing fails with section-not-found:  
   Different Digimat versions may use slightly different section titles; extend candidate titles in `eng_results_*.py`.

4. Interpolation cannot find files:  
   Ensure parsed files exist in `digimatMF_analysis/` or `digimatFE_analysis/` and match expected naming (`Analysis_b1_...`, etc.).

**中文**
1. 运行时报 `ModuleNotFoundError`：  
   需要在脚本所在目录执行（例如 `cd rve_generator` 后再 `python main.py`），因为部分导入是本地模块方式。

2. Digimat 路径报错：  
   请在 `digimatMF_runner.py` / `digimatFE_runner.py` 中改为本机安装路径。

3. ENG 提取时报 section not found：  
   不同 Digimat 版本标题可能略有差异，可在 `eng_results_*.py` 中补充候选 section 名称。

4. 分解阶段找不到文件：  
   请确认 `digimatMF_analysis/` 或 `digimatFE_analysis/` 下已存在对应命名的提取结果（如 `Analysis_b1_...`）。

---

## 11. Suggested Reproducible Run Script / 建议复现流程

**English**
A minimal end-to-end route (MF-based):

```powershell
# 1) (Optional) regenerate points/angles
cd rve_generator
python main.py

# 2) Generate MAT files
cd ..\digimatMF_analysis
python mat_generator.py

# 3) Run Digimat MF
python digimatMF_runner.py

# 4) Parse ENG results
python eng_results_mf.py

# 5) Interpolate/decompose
cd ..\orien_decomp_method
python main.py
```

**中文**
一个最小可复现流程（基于 MF）：

```powershell
# 1) （可选）重新生成点位/角度
cd rve_generator
python main.py

# 2) 生成 MAT 文件
cd ..\digimatMF_analysis
python mat_generator.py

# 3) 运行 Digimat MF
python digimatMF_runner.py

# 4) 提取 ENG 结果
python eng_results_mf.py

# 5) 执行取向分解
cd ..\orien_decomp_method
python main.py
```

---

## 12. Notes / 说明

**English**
- Current scripts are configuration-driven (edit constants at the top of each file). If needed, these can be upgraded to CLI argument mode.
- This repository currently does not include an automated test suite.

**中文**
- 当前脚本以“文件顶部参数配置”为主（修改常量即可运行）。后续可按需升级为命令行参数模式。
- 仓库当前未集成自动化测试。
