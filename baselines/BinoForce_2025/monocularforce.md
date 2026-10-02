# MonocularForce 说明

## 1. 它是什么

`MonocularForce` 是本项目为解释 BinoForce 双目增益而设置的单目力场对照方法。它不是论文中单独命名的新算法，也不是额外的基线论文；它与 `BinoForce` 使用相同的标签初始化、力场、权重、位移缩放、动态相机、warm-up 和评估窗口，唯一关键差别是：

- `MonocularForce` 只用主相机的单一视点计算重叠力；
- `BinoForce` 同时计算左眼和右眼，并对重叠比例取两者较大值：`R=max(R_left,R_right)`。

## 2. 为什么需要它

如果只比较 Baseline 和 BinoForce，最终差异同时包含“力场布局”和“双目优化”两个因素。加入 MonocularForce 后，可以形成更清楚的消融关系：

```text
Baseline
   └─ 加入连续力场更新 → MonocularForce
                              └─ 将单目重叠改成左右眼 max → BinoForce
```

因此：

- `BinoForce - MonocularForce` 近似反映双目优化本身的贡献；
- `MonocularForce - Baseline` 反映连续力场更新相对固定径向布局的贡献。

这是一种项目内的 ablation/control，不应写成“原论文提出了 MonocularForce”。

## 3. 代码路径

- 入口：`src/_pipeline.py::run_scene_experiment`
- 单目/双目共用：`dynamic_binoforce_run(..., binocular=False/True)`
- 眼睛设置：`eyes=(0.0,)` 或 `(-IPD/2,+IPD/2)`
- 其他参数完全相同：`W_REPULSE=0.02`、`W_ATTRACT=0.09`、`W_OVERLAP=0.06`、`W_LINE=0.8`、`W_CIRC=0.03`、`m=4`、`DISPLACEMENT_SCALE=0.01`。

## 4. 当前结果中的含义

当前全量汇总（55 个样本、5 个固定视角，动态评估窗口 300 帧）为：

| 方法 | OLR | LCD | DBV | quality_score |
|---|---:|---:|---:|---:|
| MonocularForce | 0.107746 | 0.011134 | 0.000806 | 20.949932 |
| BinoForce | 0.108166 | 0.010636 | 0.000783 | 21.010474 |

在这批离线数据上，BinoForce 的 DBV 和 LCD 略低于 MonocularForce，但 OLR 略高。差异较小，说明当前 proxy 几何和固定视角设置下，双目优化增益有限；这不等于否定论文结果，因为论文的真实渲染、对象遮挡、姿态朝向和用户视点都没有被完整复现。
