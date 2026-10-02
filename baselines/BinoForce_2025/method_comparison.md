# 原文方法与当前实现对照

本文以 `source/2025_Electronics_BinoForce - A Force-Based 3D Dynamic Label Layout Method Under Binocular Viewpoints.pdf` 第 4 节和当前 `src/_pipeline.py` 为依据，逐项说明原文流程、当前实现、简化点及其影响。当前项目的定位是“可运行的离线复现/代理实验”，不是原论文运行环境的完整重建。

## 1. 总体流程对照

| 阶段 | 原文 BinoForce | 当前实现 | 简化或差异 | 影响 |
|---|---|---|---|---|
| 场景输入 | 运行时 3D 场景、用户视点和动态环境 | 读取已有 Annotation JSON 与纯净 `Obj-O` OBJ | 没有实时引擎、传感器和运行时场景图 | 只能复现离线数据，不验证真实 AR/VR 延迟 |
| 标注数据 | 标签、anchor、leader line、姿态等完整 3D annotation | 仅使用 `anchor.point`、`label.center`、`label.box_size`、文本和 group id | 不维护完整姿态、pole 和 annotation 状态 | 只优化中心点，不能完全复现标签朝向和几何状态 |
| 初始化 | 标签从对应 anchor 开始 | 标签从 JSON anchor 开始 | 初始化一致；没有继承真实系统中的历史状态 | 首帧可比，连续用户运动状态未完全覆盖 |
| 相机 | 左右眼虚拟相机，随用户在场景中实时运动 | 确定性轨迹：5 秒分段、圆周运动，随机决定暂停或平移 | 用可重复轨迹替代真实用户轨迹 | 适合复现实验趋势，不代表所有交互轨迹 |
| 优化时机 | 每帧连续更新 | 每帧调用一次力布局更新，先做 warm-up，再在评估窗口统计 | 没有真实渲染帧率/线程/延迟模型 | 保留时间连续性，但不等价于实时系统 |
| 输出视角 | 由用户当前视点决定 | 固定 `main/up/down/left/right` 五个视角 | 固定视角替代任意视点 | 可比较多视角，但不能报告完整视点空间性能 |
| 评价 | 双目视觉、遮挡、重叠、leader line 和用户实验 | PCK、OLR、LCD、DBV、leader 长度 proxy、重叠对数等 | 没有真实用户研究、视线/选择时间和完整遮挡检测 | 只能做几何/投影代理评价 |

## 2. 力场与参数逐项对照

原文在第 4.2 节给出了五类力和权重。当前实现保留了可计算的核心结构，但把部分几何量改成了屏幕矩形或局部 proxy。

| 原文组件/参数 | 原文定义 | 当前实现 | 当前值 | 简化说明 |
|---|---|---|---|---|
| Repulsive force | 所有其他标签对标签产生与距离倒数相关的排斥力，并按 `n-1` 平均 | 对所有标签使用单位方向平均，不保留距离倒数 | `W_REPULSE=0.02` | 方向保留，距离衰减被去掉；近标签与远标签贡献相近 |
| Attractive force | 标签被其 anchor 吸引，力大小与 `||l_i-a_i||` 成正比 | 将标签拉向数据驱动的 `circular_radius`，不是直接拉回 anchor | `W_ATTRACT=0.09` | 保留“不要离 anchor 太远”的意图，但平衡点和原文公式不同 |
| Overlapping force | 计算标签-标签或标签-物体重叠面积比；双目取左右眼较大值 | 使用左右眼投影矩形的重叠比例；物体使用 anchor 周围局部立方体 proxy | `W_OVERLAP=0.06`, `m=4.0` | 双目 max 和 `m=4` 保留；真实物体 mask 改为 proxy/矩形 |
| Leader-line intersection | 用交点两侧三角形面积计算 `D_ij`，并按局部法向决定方向 | 保留投影线段相交、三角形面积比和法向方向 | `W_LINE=0.8` | 公式结构基本保留；仍是 2D 投影相交代理 |
| Circular force | 原文引用 Pick 等人的圆形布局力 | 用场景中心到标签的径向距离，将标签拉到 `circular_radius` | `W_CIRC=0.03` | 采用确定性 radial heuristic，不是原方法的完整实现 |
| Resultant force | `w1..w5` 加权合力 | 相同五个权重组合 | `(0.02,0.09,0.06,0.8,0.03)` | 权重与论文一致；单项力的几何定义不完全一致 |
| Displacement | 每帧由合力得到位移，缩放经验系数 `0.01` | `labels += forces * 0.01` | `DISPLACEMENT_SCALE=0.01` | 保留 |
| Label orientation | 每帧调整为面向用户 | 当前不维护旋转，只用 camera-facing bbox 估算标签矩形 | 无显式姿态参数 | 不能评价真实文字朝向/旋转稳定性 |
| Binocular baseline | 左右虚拟相机间距 6.4 cm | `IPD_METERS=0.064`，左右眼偏移为 `±0.032 m` | `0.064 m` | 保留数值；视点模型仍是简化相机 |

## 3. 数据、几何和评估的简化

### 3.1 场景几何

- 原文使用可交互的完整 3D 场景和对象几何。
- 当前布局力不直接使用完整 OBJ 表面，而是对每个 anchor 建立局部立方体 proxy：`eps=max(0.08*||size||, 0.035)`。
- 评价和预览会读取纯净 `Obj-O` 的主体顶点；解析到 `label_*` 或 `leader_*` 组时停止，从而避免把已有标签当成物体。
- 原文提到的 bounding sphere / 真实对象体积，在当前代码中用 anchor 集合和 OBJ/anchor 的 axis-aligned proxy 近似；因此 `scene_center`、`scene_radius` 和 `circular_radius` 都是数据驱动的近似量。

### 3.2 时间流程

原文的连续更新被当前实现拆成两个离线阶段：

1. 标签从 anchor 初始化。
2. 相机按 30 FPS 的确定性轨迹运动；每 5 秒为一个段，段内可能暂停或沿地面平移。
3. 每帧执行一次 `force_layout(..., iterations=1)`。
4. 前 `warmup_frames` 帧只用于让布局稳定；后 `eval_frames` 帧计算指标均值。
5. 默认候选 warm-up 为 `300,500,800`，在 validation subset 上选择；当前生成结果选择了 `800`，并非在全量结果上挑最优值。

原文没有公开可直接对应的迭代次数；因此 `warmup_frames` 是复现实验设计参数，不应解释为论文原始迭代次数。

### 3.3 评估指标

当前使用的主要指标如下：

- `PCK_005/PCK_010`：与 manual center 的归一化屏幕距离阈值命中率。
- `OLR`：标签与标签/物体投影矩形重叠面积占标签面积的比率。
- `LCD`：leader line 投影交叉程度的平均值。
- `DBV`：左右眼每标签重叠面积差的归一化平均值。
- `avg_leader_length`：屏幕空间 leader 长度除以图像对角线的 proxy。
- `overlap_pairs`、`occluded_points`、`intersections`：离线矩形/线段统计量。
- `quality_score`：项目自定义综合分数，不是论文报告的原始统计量。

这些指标可以比较本项目内的 Baseline、MonocularForce、BinoForce，但不能直接宣称等价于原论文的用户实验或真实遮挡结果。

## 4. 结论

当前实现保留了最关键的可验证骨架：3D 标签中心、连续逐帧更新、双目左右眼、五类力、论文权重、6.4 cm IPD、动态相机和 DBV/OLR/LCD 代理评价。主要被简化的部分是运行环境、完整 annotation 姿态、真实对象遮挡、力的精确几何定义、实时交互状态和用户研究。

因此，本项目适合回答“不同方法在同一批离线场景上的相对趋势”，不适合回答“已完整复现论文系统或已达到论文用户实验效果”。
