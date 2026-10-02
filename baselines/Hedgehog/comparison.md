# Hedgehog Labeling 复现实验记录

## 复现依据

- 论文方法是纯几何/优化方法，没有训练阶段；本复现不引入学习模型。
- 3D pole 方向：按论文从物体包围球中心指向 anchor point 的径向向量确定。
- 1D hedgehog：只允许 annotation 沿 pole 改变长度。
- 3D hedgehog：在 pole 长度之外，允许 annotation 在其局部图像平面 X/Y 方向移动，并把该位移限制在 annotation size 内。
- plane 方法：使用与当前视平面平行、在 screen-aligned bounding box 中等距放置的平面，并把 label 分配到最近平面。

## 参数说明

- `plane_count=3`：论文说明该值由用户在运行时设置，并在 Figure 7 示例中使用 3 个平面；本复现默认采用 3。
- `iterations=160`：数值优化迭代上限，只影响收敛时间，不改变论文约束。
- 相机视角：采用 `regenerate_layout_assets(1).py` 对应工具链中的 multiview camera 设定。

## 数据与输出

- 数据：`../data/Layout`，共 55 个 3D 标注样本，每个样本评估 main/up/down/left/right 五个视角。
- 布局 JSON：`results/layouts/<Category>/<Sample>/<View>/<Method>.json`。
- 预览图：`results/previews/*.png`，索引页面为 `results/layout_preview.html`；manual 列使用数据集 Mutiviews 原图，复现方法列使用 pyrender 渲染同一 OBJ 主体后叠加 label。
- 指标：在与预览渲染相同的相机状态下重算；PCK 以 data 中 manual label center 为参考，OLR/LCD 按屏幕投影计算，avg_leader_length 按图像对角线归一化。

## 平均指标

| method | PCK@0.05 ↑ | PCK@0.10 ↑ | OLR ↓ | LCD ↓ | avg_leader_length ↓ | quality_score ↑ |
|---|---:|---:|---:|---:|---:|---:|
| hedgehog_1d | 0.082 | 0.333 | 0.098 | 0.000 | 0.069 | 13.300 |
| hedgehog_3d | 0.077 | 0.343 | 0.029 | 0.000 | 0.076 | 14.983 |
| manual | 1.000 | 1.000 | 0.015 | 0.001 | 0.164 | 99.621 |
| plane | 0.078 | 0.346 | 0.035 | 0.000 | 0.075 | 14.967 |