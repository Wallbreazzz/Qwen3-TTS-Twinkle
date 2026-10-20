# TTS 微调学习项目

在昇腾 910B NPU 上，用 **Twinkle 框架 + LoRA SFT** 微调开源 TTS 模型（`Qwen3-TTS-12Hz-1.7B-Base`、`MOSS-TTS-Local-Transformer-v1.5`），跑通"数据→训练→评估→部署"闭环，并为 Twinkle 新增 TTS 训练组件回流本方框架。

## 技术栈
`torch-npu + CANN` → **Twinkle**（`TransformersModel + add_adapter LoRA + DeviceMesh + kernelize`）→ 官方模型 tokenizer
（昇腾无 CUDA，弃用 unsloth/bitsandbytes/flash-attn2；框架层 Twinkle 由本方主导、客户也将用）

## 入口（每个新会话第一件事）
**读 [`新会话必读.md`](新会话必读.md)** —— 项目目的、文件地图、可用 skill、当前进度、下一会话起点都在那。

- 协作规则与硬约束：[`AGENTS.md`](AGENTS.md)
- 术语：[`docs/01-术语表与基础概念.md`](docs/01-术语表与基础概念.md)
- 路线图：[`docs/00-总体方案.md`](docs/00-总体方案.md)
