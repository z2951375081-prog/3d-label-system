# 独立 val 六图视觉评分与跨轮趋势

> 2026-09-19 更新：十四维评分中，`manual_style_similarity`、`spatial_balance`、`visual_hierarchy`、`composition_harmony`、`overall` 五维形成美学综合分；其余九维只作安全诊断。穿模、遮挡、越界、文字和引导线由确定性能量与安全门控负责。具体权重和阈值见 [美学偏好与几何安全职责分离](美学偏好与几何安全职责分离_2026-09-19.md)。本文后文旧的十二维混合公式不再用于新运行。

## 输入与隔离

- 训练对只允许 manifest 中的 train 样本；评分凭据只有 train 生成，不从 val/test 产生。
- 在收集偏好前，对 0 轮无奖励基线生成 val 布局。每次累计训练后，将候选奖励模型和 SHA-256 写入 `llm_preference_checkpoints/<run_id>/round_<n>.json`，并以**该文件**而非可能被后续轮改写的活动模型生成同一批 val 布局。
- val 样本由 manifest 确定；默认 11 个，实验可指定较少样本做冒烟测试，但只有同一批完整 val11 的 0/1/2/4/8 轮有资格被判定为充分趋势证据。
- 五视角采用项目数据集校准过的统一相机协议，每次原始干净 OBJ 图加布局的主/右/左/俯/仰共六图。人工调整后 JSON 只用于固定标签契约、人工风格相似度和独立评分，不把最终中心/尺寸作为奖励网络推理特征。

## 请求预算与留证

每次运行预计六图外部请求：连接测试一次（若用户执行） + `train 样本数 × 候选数 × 轮数` + `val 样本数 × (轮数 + 1)`。例如 3 个 train 样本、4 候选、8 轮、val11：不含连接测试为 `96 + 99 = 195` 次。每次请求处理六张图片；正式运行前应确认费用和模型权限。

服务端 `/api/generate` 在 val 下按指定 run ID 和轮次加载冻结检查点，验证 SHA-256 并给出一次生成凭据。`/api/score` 在调用外部接口之前核验布局、指标、样本和检查点凭据；持久化的逐样本 val JSON 保存响应 ID、十四维分数、几何代理分、检查点哈希和文本理由，不保存图片或 API Key。`/api/llm-visual-validation/finalize` 要求全部预期 val 样本都已真实评分，计算十四维均值及五维美学综合分，写入 `experiments/llm_visual_validation_<run_id>.json`。不把 val 评价写入 `preferences.jsonl`。

美学综合分为 `0.30 × overall + 0.25 × composition_harmony + 0.20 × visual_hierarchy + 0.15 × spatial_balance + 0.10 × manual_style_similarity`。其余九个安全维度不进入偏好胜者加权，只用于诊断和防退化。只有已通过 val 美学提升与确定性几何安全联合门控的检查点才能激活；仅当所有可选检查点都覆盖同一个完整 val11、使用同一评分模型且冻结哈希一致时，最终 test 才按视觉美学综合分选轮。1～10 个 val 样本只允许做流程冒烟，选轮回退到已门控的候选自身美学代理，不能称为视觉选模。如果没有已接受检查点，test 仅供诊断，不能激活拒绝模型。`experiments/llm_preference_convergence_<run_id>.json` 独立列出视觉美学分、安全诊断和几何代理分。判断“先升后稳”的预设规则：相对基线，某已训练轮美学综合分至少提高 0.05，九个安全维度每维至多下降 0.15；第 4 与第 8 轮美学综合分差至多 0.1 且五个美学维度平均绝对变化至多 0.15。未齐备五个检查点或 val11 样本时结论为证据不足。

在本机网页输入外部模型 API URL、模型 ID 与新 API Key，通过“测试六图与 JSON”后再启动；不要把 Key 发到聊天或项目文件。`npm run test:scoring` 用本地模拟视觉服务器验证多轮协议，**模拟评分不是正式模型效果数据**。

正式实验预检命令：

```powershell
npm run preflight:llm-preference -- --samples 3 --candidates 4 --rounds 8 --visualValSamples 11
```

预检报告保存为 `experiments/llm_preference_preflight.json`。只有 `ready=true` 才应发起外部请求；`external_visual_connection` 会检查当前 Node 进程内存中的 Key 是否已经通过六图和十四维 JSON 测试，但不会输出或保存 Key。本地可先生成最多 `候选数 × 10` 个布局筛选互相可比的安全候选，外部只评分最多 4 个，因此 3 个 train 样本、4 个安全候选、8 轮、完整 val11 的外部请求上限仍为 195 次、1170 张渲染图，另加一次连接测试请求。

正式运行完成后执行：

```powershell
npm run analyze:preference-convergence -- --runId <本次运行ID>
```

该命令只读取已落盘的真实 val 六图评分和不可变检查点，不重新训练模型、不调用外部接口，也不把几何代理分冒充视觉分。旧版几何分析报告会另存为 `llm_preference_convergence_legacy_geometry_<run_id>.json`。分析器必须同时验证 0/1/2/4/8 五个点、同一 val11 样本、同一评分模型和检查点 SHA-256，才可能输出 `evidence_sufficient=true`。
