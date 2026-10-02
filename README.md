# OBJ 3D Label Studio

本项目是面向 3D 外部标签布局研究的本地实验系统。当前主线为：DGCNN 从 clean OBJ 表面提取 64D 几何，锚点表面 patch 构建局部坐标系，51D 局部三维标签特征经残差 FNN、Anchor–Label 关系类型感知异构 GNN、全局 Transformer 和人类风格 MoE 解码；新增轮廓驱动的 spherical / rectangular / surround 三风格 MoE 初始化与路由软目标；五视角只负责安全评价与确定性优化，本地 Qwen3-VL 只在安全候选中做美学重排。

当前版本：2026-09-21。

## 当前模型架构

基础布局模型采用 DGCNN–局部坐标–异构关系 GNN–Transformer–人类风格 MoE，而不是逐标签独立 MLP：

- 一个物体的全部固定标签组成一张图；
- 每个固定标签对应一个 label 节点，每个锚点表面 patch 对应固定 anchor 几何上下文；
- anchor→label 为 18 维独立关系通道，包含局部坐标、锚点法向、引导线方向/长度、曲率、patch 尺度、标签尺寸与部件关系；
- label→label 为 10 维独立关系通道，包含三维相对位置、距离、尺寸比例、方向夹角、共享 source object 与 target group；
- 51→64 节点编码；
- 一层 64→128→64 的残差 FNN，在 GNN 消息传递前提取节点内部信息；
- 从 clean OBJ 按三角形面积确定性采样 1024 个表面点，两层动态 kNN EdgeConv 输出 64 维物体几何特征；
- 对每个锚点在半径 patch 内做加权 PCA 和面积加权法向平均，得到 `t1,t2,n`；高曲率或低法向稳定度时自动扩大 patch，并保留对称物体的等价切向方向；
- 64D 标签与 64D OBJ 几何拼接成 128D，经 128→64 融合层进入异构关系图；五视角像素不进入初始三维坐标生成器；
- 两层关系消息传递分别学习 anchor→label 与 label→label 参数；
- 一层缩放点积自注意力与两倍宽度 FFN，提取整组标签的全局信息；
- 可学习逐节点 Softmax 路由器融合 compact、balanced、spacious、long-text 等人类风格专家；
- 每个专家采用 64→32→6 输出头，软加权后预测局部 `u,v,normal-distance` 与面板三轴尺寸对数比例；
- 五视角损失主视角权重 0.40，其余各 0.15，并加入 silhouette 交叠、深度遮挡、穿模、字体清晰度、逐视角留白分布、worst-view、CVaR 与双目一致性。

数据集人工标注固定标签数量、ID、文字、锚点和语义关联。人工最终中心、尺寸以及画有人工标签的多视角 PNG 均不进入基础图网络输入。训练目标为确定性五视角安全优化器产生的布局教师，因此基础图网络学习物体表面、无标签视图、标签关系几何与安全初始化。

详细说明见 docs/本地Qwen与关系图偏好架构_2026-09-20.md。

## 本地 Qwen3-VL 偏好训练

视觉偏好评分仅使用本机 Ollama：

- 地址：http://127.0.0.1:11434
- 模型：qwen3-vl:4b-instruct
- 接口：/api/chat
- 输入：原始模型图 + 主、右、左、上、下五视角，共六图
- 输出：JSON Schema 约束的十四维 1–5 分
- 上下文：32768

项目不再包含 OpenAI Responses、Chat Completions、第三方评分地址或密钥输入。服务端只接受 localhost、127.0.0.1 或 ::1 的 Ollama 地址，图片不会上传到外部服务。

十四维评分分工如下：

- 九个安全诊断维度：文字清晰、覆盖、标签重叠、物体遮挡、穿模、引导线、多视角、双目、尺寸一致性；
- 五个美学维度：人工风格相似、空间平衡、视觉层级、构图和谐、整体美感。

五个美学维度形成 chosen/rejected，训练 12→32→1 pairwise 奖励网络。奖励网络只在确定性安全门控通过的候选中重排，不取代图网络，也不能绕过几何安全限制。

## 安装与启动

先确认 Ollama 和模型：

~~~powershell
ollama list
ollama run qwen3-vl:4b-instruct
~~~

启动项目：

~~~powershell
node server.mjs
~~~

然后打开 http://127.0.0.1:5173/。也可双击 启动3D标签系统.bat。

项目使用原生 WebGL，不依赖外部 CDN。

## 训练与验证

训练 v10 锚点局部坐标异构图候选：

~~~powershell
npm run train:layout:v10 -- --epochs 80 --learningRate 0.002 --directionWeight 1.25 --viewWeight 0.25 --worstViewWeight 2 --cvarViewWeight 1 --textClarityWeight 3 --leaderCrossingWeight 3.5 --hiddenDim 64 --messageLayers 2 --transformerLayers 1 --expertCount 4 --seed 17
~~~

候选写入 `experiments/layout_model_v10_anchor_frame_candidate.json`，不会由训练脚本直接覆盖活动模型。训练报告记录输入 FNN、融合层、两类关系参数、Transformer、MoE 路由器与专家的真实参数更新。使用 val11 搜索推理融合比例并保留安全诊断：

~~~powershell
npm run select:layout:v10 -- --previous experiments/layout_model.json
~~~

只有验证门控通过后才允许备份并替换 experiments/layout_model.json；test11 只用于最终确认。

偏好学习：

~~~powershell
npm run preflight:llm-preference -- --samples 3 --candidates 4 --rounds 8 --visualValSamples 11
npm run run:llm-preference -- --samples 3 --candidates 4 --rounds 8 --visualValSamples 11
~~~

偏好训练只读取 manifest 中的 train 样本；val 用于模型选择，test 只作最终评估。

## 测试

~~~powershell
npm run test:scoring
npm run test:moe-styles
npm run test:preference
npm run test:layout-provenance
npm run audit:fixed-labels
~~~

其中 test:scoring 验证本地回环地址限制、六图 Ollama 请求和十四维结构化结果解析。

## 主要文件

- lib/cv-feature-encoder.mjs：clean OBJ 1024 点表面采样、两层 EdgeConv、无标签五视角渲染和冻结轻量 CNN。
- lib/anchor-frame-features.mjs：加权 PCA 局部表面坐标系、曲率/法向稳定性、自适应 patch 与边特征。
- lib/heterogeneous-layout-graph.mjs：anchor/label 节点契约、18D 与 10D 关系类型、局部坐标监督和解码。
- lib/layout-model.mjs：128D 纯三维融合、关系类型感知 GNN、Transformer、人类风格 MoE 推理和旧版本兼容读取。
- scripts/train-3d-human-style-layout-model.mjs：v9/v10 完整训练、worst-view/CVaR、引导线方向/交叉风险与字体清晰度损失。长训练每 10 分钟检查一次。
- lib/ollama-vision-adapter.mjs：本地 Qwen3-VL 六图评分适配器。
- scripts/train-preference-model.mjs：十二维美学 pairwise 奖励训练。
- lib/moe-layout-styles.mjs：从 OBJ 主视角轮廓点云和锚点分布中计算 spherical / rectangular / surround 三风格路由，并生成球状环绕、矩形边界、锚点围绕三类初始化布局。
- lib/layout-optimizer.mjs：五视角确定性安全能量与模拟退火；可通过 layoutStyle/styleExpert 接入三风格 MoE 初始化后再优化。
- server.mjs：本地服务、模型门控和实验凭据。
- public/app.js：浏览器工作台与偏好实验自动化。

## MoE1 三风格轮廓布局

新增的 MoE1 路由先从 clean OBJ 提取主视角轮廓点云和凸包，再结合三维长宽比、轮廓规整度、锚点角度熵和径向一致性，归一化得到 spherical（球状/环状）、rectangular（矩形边界）和 surround（自由围绕）三类概率。

在推理或伪标签生成时，可调用：

~~~js
optimizeLabels(labels, bounds, { layoutStyle: 'auto', geometry, optimizer: 'annealing' })
~~~

其中 spherical 会按轮廓外椭圆弧分布标签，rectangular 会沿物体投影包围矩形四边分布，surround 会按锚点方向围绕物体但不强制对齐。可用下面的命令审计 55 个样本的风格路由，并为 train33 生成无监督伪标签：

~~~powershell
npm run audit:moe-styles
npm run generate:moe-pseudolabels -- --iterations 120
npm run prepare:moe-style-llm
npm run run:moe-style-llm
npm run analyze:moe-style-llm-prior
npm run generate:moe-layout-llm-candidates -- --samples 2
npm run render:moe-layout-llm-candidates
npm run score:moe-layout-llm-candidates
npm run generate:moe-expert-layouts -- --split train --iterations 120
npm run select:moe-expert
npm run train:layout:v10:expert-selection
~~~

伪标签会写入 `experiments/moe_unsupervised_pseudolabels_train.json`。MDPO 候选扰动已包含增大标签间隙、改变引导线长度、按锚点局部坐标调整弧向均匀度，并在主视角做视野回缩限制。已用本地 Qwen 对 55 个样本完成五视角风格初筛，结果写入 `experiments/moe_style_llm_scores.json`；`npm run analyze:moe-style-llm-prior` 显示 LLM 与几何先验差异较大，因此仍需人工评分模板完成显著性验证。v10 训练可通过 `npm run train:layout:v10:pseudo` 或手动传入 `--targetSource pseudo --pseudoLabels experiments/moe_unsupervised_pseudolabels_train.json` 使用这些模拟退火标签替代 train33 人工中心作为监督目标；当前诊断候选保存在 `experiments/layout_model_v10_moe_pseudo_candidate_diagnostic.json`；val11/test11 仍默认保留人工标注用于选择与最终确认。可视化界面见 `http://127.0.0.1:5173/moe-dashboard.html`，页面可以选择全部 test11 样本查看三个专家布局、最终选择、router 概率和平均指标；详细设计见 `docs/MoE1三风格轮廓布局实施记录_2026-09-29.md`。

## 数据和实验边界

数据集共 55 个样本，按类别分层划分为 train33、val11、test11。固定标签契约要求数量、顺序、ID、文字、锚点和 sourceObjs/targetGroups 在候选、推理和优化阶段保持一致。

BinoForce、Hedgehog 等复现结果仍保存在对应目录和 experiments/comparisons 中，用于统一相机协议下的离线对照。历史实验 JSON 可用于审计，但不会被当前本地 Qwen 训练链路自动读取。

## 最新模型与复现方法的统一指标

运行下面的命令，会使用活动 `experiments/layout_model.json` 对全部 55 个样本重新推理，并在同一个 750×500、固定世界原点、50 mm 五视角相机协议下评价最新 v10、BinoForce、Hedgehog 1D/3D 和人工优化标注：

~~~powershell
npm run evaluate:latest-comparison
npm run test:reproduction-metrics
~~~

统一指标为 `PCK_005`、`PCK_010`、`OLR`、`LCD`、`DBV`、`avg_leader_length`、`overlap_pairs`、`occluded_points`、`intersections` 和安全质量分 `quality_score`。Hedgehog 没有双目复现，因此其 DBV 为 N/A，而不是 0；人工标注自身是 PCK 参考，所以 PCK=1 只表示参考自一致。`quality_score` 统一采用 v3 安全优先公式（PCK 合计 25%，重叠、遮挡、交叉与引导线合计 75%），不是原论文美学分。

结果写入 `experiments/comparisons/latest_reproduction_metrics/`：

- `comparison.json`：协议、活动模型 SHA、逐样本/逐视角结果及 train33、val11、test11、all55 汇总；
- `rows.csv`：每个样本、每种方法的五视角均值；
- `view_rows.csv`：55×5×5=1375 条逐视角结果；
- `summary.csv` 和 `main_summary.csv`：划分级五视角均值与主视角均值。

浏览器默认打开 test11，并明确显示样本所属 train/val/test。界面只展示活动模型结果、v10 架构、五视角、统一方法表和人工优化标注。BinoForce 表中使用保存的最终静态布局快照重新投影，不能与其原复现 CSV 的 300 帧动态均值混为一谈。Hedgehog 最终表从保存的布局 JSON 重新计算；旧 `Hedgehog/results/hedgehog_results.csv` 有少量行与最终 JSON 不一致，仅保留作历史记录。








## GitHub 与学校服务器训练

仓库地址：`https://github.com/z2951375081-prog/3d-label-system.git`

在学校服务器上首次使用时：

```bash
git clone https://github.com/z2951375081-prog/3d-label-system.git
cd 3d-label-system
git lfs install
git lfs pull
npm install
npm run dataset:validate
```

训练前建议使用 `tmux`，避免 SSH 断开导致任务停止：

```bash
tmux new -s thesis-train
npm run train:layout:v10 -- --epochs 80 --learningRate 0.002 --directionWeight 1.25 --viewWeight 0.25 --worstViewWeight 2 --cvarViewWeight 1 --textClarityWeight 3 --leaderCrossingWeight 3.5 --hiddenDim 64 --messageLayers 2 --transformerLayers 1 --expertCount 4 --seed 17
```

暂时离开训练窗口：按 `Ctrl+B`，再按 `D`。重新进入：

```bash
tmux attach -t thesis-train
```

OBJ 三维数据通过 Git LFS 下载；`experiments/` 中的训练缓存、浏览器缓存和大规模中间结果不提交到 GitHub，会在服务器训练时重新生成。

> 说明：默认的 55 个样本位于 `data/Layout/`，会随仓库通过 Git LFS 下载。扩展的 `data/Layout_200train_50test/` 数据集体积较大且与默认数据存在重复，未放入 GitHub；如需 200-train/50-test 训练，请将该目录单独上传到学校服务器的项目目录中。

## V10 共享图网络 + 三专家 Style-Adapter MoE

已加入不依赖本地训练的 V10 代码入口：共享锚点帧异构图网络先输出共享表示，再注入三组 style embedding 和轻量 residual Adapter；三个专家均输出 6D 局部布局，Router 支持软混合、显式权重混合和指定单一风格。后处理提供带风格保持项的模拟退火，并在 `main/right/left/up/down` 五视角安全门控下接受或拒绝候选。

本地只运行：

```bash
npm run check:v10:style-adapter-moe
npm run test:v10:style-adapter-pipeline
```

不要在本地执行训练命令。学校服务器拉取 GitHub 后，按 `docs/V10风格AdapterMoE训练说明.md` 中的 `npm run train:layout:v10:style-adapter-moe` 启动训练。
