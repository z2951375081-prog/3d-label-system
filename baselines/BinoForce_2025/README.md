# BinoForce 2025 离线复现整理版

本项目复现 Zheng et al. 的 **BinoForce: A Force-Based 3D Dynamic Label Layout Method Under Binocular Viewpoints**（Electronics 2025）中的可验证部分，并在同一数据与相机条件下比较 Baseline、MonocularForce、BinoForce，以及已导入的 Hedgehog/manual/plane 结果。

## 先看什么

1. [docs/method_comparison.md](docs/method_comparison.md)：原文流程、参数与当前实现的逐项差异，以及所有主要简化。
2. [docs/monocularforce.md](docs/monocularforce.md)：MonocularForce 的定义、代码路径和消融意义。
3. [docs/paper_method_draft.md](docs/paper_method_draft.md)：可直接继续润色到论文中的英文方法段落。
4. [BinoForce_2025_实验总结.xlsx](outputs/20260727_experiment_summary/BinoForce_2025_实验总结.xlsx)：实验参数、逐行结果、方法/类别汇总、validation sweep、DBV 对照和指标字典。
5. [docs/reproduction_notes.md](docs/reproduction_notes.md)：本次复现的运行边界和结果规模。
6. `results/report.html`：已有 HTML 可视化报告。

## 当前实验设置

- 场景数：55
- 固定视角：`main`、`up`、`down`、`left`、`right`
- 动态相机：30 FPS；每 5 秒一个轨迹段；确定性圆周运动并加入暂停/地面平移段
- warm-up：在 validation subset 上从 `300,500,800` 中选择，当前为 `800` 帧
- 评估窗口：300 帧
- 双目基线：左右眼间距 0.064 m（6.4 cm）
- 原文力权重：`W_REPULSE=0.02`、`W_ATTRACT=0.09`、`W_OVERLAP=0.06`、`W_LINE=0.8`、`W_CIRC=0.03`
- 重叠放大常数：`m=4.0`
- 位移缩放：`0.01`

## 运行

```powershell
& 'C:\Users\chenyv\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\src\reproduce_binoforce2025.py --data ..\data --out .\results --hedgehog-results ..\Hedgehog\results\hedgehog_results.csv --eval-frames 300 --sweep-candidates 300,500,800 --fps 30
```

重新生成 Excel：

```powershell
& 'C:\Users\chenyv\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\tools\build_experiment_excel.py
```

## 目录结构

```text
BinoForce_2025/
├─ README.md                         项目入口
├─ docs/                             方法对照、说明和复现边界
│  └─ archive/                       原始办公文档/历史记录归档
├─ source/                           原论文 PDF 与文本提取版
├─ src/                              主实现、数据加载、评价、报告和可视化
├─ baseline/                         单独的 Baseline 复现脚本及其结果
├─ results/                          主实验结果、报告、图像和 Excel
└─ tools/                            可重复运行的结果整理脚本
```

布局力使用 anchor 周围的局部对象 proxy；评价和预览使用跳过 label/leader group 的纯净 OBJ 主体。当前实现不包含论文原始实时引擎、完整 annotation 姿态、真实遮挡 mask、VR 设备延迟或用户研究，因此结果应解释为离线相对比较，而不是完整系统复现。

## 代码结构

- `src/reproduce_binoforce2025.py`：命令行入口。
- `src/_pipeline.py`：核心数据结构、相机、力布局、动态运行、指标和报告逻辑。
- `src/data_loader.py`、`src/cameras.py`、`src/binoforce.py`、`src/evaluate.py`、`src/reporting.py`、`src/visualize.py`：兼容性 facade/import 模块。
- `tools/build_experiment_excel.py`：将 CSV/JSON/汇总结果生成格式化 Excel。

`MonocularForce` 是项目内的单目 ablation/control：它与 BinoForce 共用同一力场，只把左右眼重叠评价改为主相机单视点，不是论文中单独提出的第三种方法。


