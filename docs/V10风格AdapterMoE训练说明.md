# V10 Style-Adapter MoE 训练入口

本次新增的模型入口对应以下链路：

```text
共享 V10 锚点帧异构图网络
        ↓
style embedding（radial_ring / box_sides / anchor_adaptive）
        ↓
三个轻量 residual Adapter
        ↓
每个专家输出 6D 局部布局（u、v、normal、log-size-x/y/z）
        ↓
Router 输出逐标签 softmax 权重
        ↓
router 混合 / 显式 styleWeights 混合 / 指定 styleExpert
        ↓
可选带 style preservation loss 的模拟退火
        ↓
main、right、left、up、down 五视角安全门控
```

## 代码位置

- `lib/style-adapter-moe.mjs`：style embedding、三个 Adapter、6D expert head、Router 和三种融合方式。
- `lib/layout-model.mjs`：将该 MoE 接入 V10 推理；可传 `styleExpert`、`styleWeights` 或 `forcedStyle`。
- `lib/style-preserving-annealer.mjs`：风格保持项、模拟退火和五视角 hard safety gate。
- `scripts/train-3d-human-style-layout-model.mjs`：服务器端训练入口，保留 train/val/test 隔离，test 不参与 checkpoint 选择。
- `scripts/test-v10-style-adapter-pipeline.mjs`：不训练的结构契约和安全门控测试。

## 学校服务器训练

本地只做结构检查，不启动训练。服务器拉取仓库后：

```bash
npm install
npm run check:v10:style-adapter-moe
npm run test:v10:style-adapter-pipeline
npm run dataset:validate
```

启动三专家 V10 训练：

```bash
npm run train:layout:v10:style-adapter-moe -- \
  --epochs 80 \
  --learningRate 0.002 \
  --hiddenDim 64 \
  --messageLayers 2 \
  --transformerLayers 1 \
  --styleEmbeddingDim 8 \
  --styleAdapterDim 16 \
  --styleWeight 0.2 \
  --directionWeight 1.25 \
  --viewWeight 0.25 \
  --worstViewWeight 2 \
  --cvarViewWeight 1 \
  --stereoWeight 1 \
  --textClarityWeight 3 \
  --leaderCrossingWeight 3.5 \
  --styleLabels experiments/gpt_style_judgment_main_view.json \
  --output experiments/layout_model_v10_style_adapter_moe_candidate.json \
  --report experiments/layout_training_v10_style_adapter_moe_report.json
```

训练脚本会生成候选模型和报告，但不会自动覆盖 `experiments/layout_model.json`。完成后仍需运行现有的 V10 validation gate，再决定是否激活模型。

## 推理时的风格控制

默认使用 Router 软混合；也可在 `applyLayoutModel` 的 options 中使用：

```js
{ styleExpert: 'radial_ring' } // 指定单一风格
{ styleWeights: { radial_ring: 0.5, box_sides: 0.3, anchor_adaptive: 0.2 } }
```

如果要启用后处理安全门控，需要把五视角 depth grids 传给 `applyLayoutModel`：

```js
{
  postProcess: 'style_preserving_annealing',
  depthGrids,
  hardSafetyGate: true,
  styleWeight: 1
}
```

安全门未通过时，结果会带有 `five_view_safe: false`，不能当作已验证布局发布。
