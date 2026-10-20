# AGENTS.md

本项目用于学习 TTS 模型微调。给 AI 助手的**协作约定与硬规则**（本文件的约束是权威来源）。

## 项目背景
- 在昇腾 910B NPU 上，用 Twinkle 框架 + LoRA SFT 微调 `Qwen3-TTS-12Hz-1.7B-Base` 与 `MOSS-TTS-Local-Transformer-v1.5`。
- 用户是"模型微调"和"语音"领域新手，软件工程经验丰富。
- 框架层 Twinkle 由本方（招商银行开源技术团队等）主导，客户也将使用；核心扩展点是为 Twinkle 新增 TTS 训练组件并回流本方框架。

## 关键技术约束（务必遵守，违反=返工）
- **硬件是昇腾 NPU，不是 NVIDIA GPU**：无 CUDA/Triton。
- **框架层固定为 Twinkle**（`github.com/modelscope/twinkle`）：用 `TransformersModel + add_adapter_to_model(LoRA) + DeviceMesh + kernelize`。不要换回裸 HF 四件套或换 MindSpeed-MM/unsloth/LLaMA-Factory——框架层是本方与客户商定的根本决策，要改先与用户确认。
- **禁止建议** `unsloth`、`bitsandbytes`(QLoRA)、`flash-attention-2`、`xformers`（均 CUDA-only，昇腾不可用）。
- **不要推荐** `LLaMA-Factory/Axolotl` 作训练主干（不支持 TTS 的多码本/局部 Transformer 训练格式）。
- 数值精度用 **bf16**（昇腾原生支持），不用 4bit。
- 用户倾向**单卡**训练，哪怕时间翻倍；`DeviceMesh` 用 `fsdp_size=1`，不擅自引入分布式。
- MindSpeed-MM 仅作备选：仅当将来目标改成 MOSS-TTS-Delay 或要多卡全量SFT 才考虑。

## 协作风格约定
- **写方案/文档时**：每个步骤、每个专业词汇都要用大白话解释清楚（用户是新手）。可引用 `docs/01-术语表与基础概念.md` 而非重复定义。
- **写 skill/代码时**：可用专业语言，不必逐词解释。
- 遇到要修改"技术栈/框架层"这种根本决策时，先与用户确认，不要擅自换栈。

## 工作流程（阶段推进）
1. **先读 `新会话必读.md`**（每个新会话第一件事）→ `docs/00-总体方案.md` → `docs/01-术语表与基础概念.md` 建立背景。
2. 训练相关任务优先加载对应 skill；部署相关任务加载本地已装的 `vllm-omni-*` skills。
3. 阶段顺序：阶段0 环境(Twinkle+昇腾) → 阶段1 数据 → 阶段2 新增TTS组件+训练 → 阶段3 评估部署。

## 文件/skill 地图与当前状态
详见 `新会话必读.md`（单一来源，此处不重复，避免信息漂移）。简言之：准备完成，下一会话起点=阶段0。
