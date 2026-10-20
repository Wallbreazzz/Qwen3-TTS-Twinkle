"""Qwen3-TTS 自定义 forward 编排（W-2 附加 / W-4）。

移植自官方 `Qwen3-TTS/finetuning/sft_12hz.py` 的训练步 4 步手动流程（见 docs/05 第 5 节）。
Twinkle 标准 `model.forward_backward(inputs=batch, task='causal_lm')` 走"单次 model(**inputs)→loss"
路径，管不住 Qwen3-TTS 的：① speaker_encoder 前置 ② 手动组装 inputs_embeds ③ 2 段 forward。
故需自定义编排，本文件给出逻辑实现；Twinkle 集成方式见模块末"集成选项"。

W-4 变更：编排不再自己算总 loss，改为产出**中间态 dict**（talker_loss / sub_talker_logits /
sub_talker_labels / target_speaker_embedding），总 loss 由 Qwen3TTSLoss 从中间态算（职责清晰：
编排管前向，Loss 管 loss）。sub_talker 段取 forward_sub_talker_finetune 返回的 logits、
忽略其返回的 loss（继承的 ForCausalLMLoss 对 [N,15,V] 码本轴有 off-by-one，见 loss.py docstring）。

⚠️ 需要 Qwen3-TTS 真实模型权重才能跑（CPU 跑不动 forward），本文件为逻辑骨架 + W-5/W-6 在 NPU 验证。
"""

from typing import Dict, Optional
import torch


def qwen3_tts_forward_step(
    model,
    batch: Dict[str, torch.Tensor],
    target_speaker_embedding: Optional[torch.Tensor] = None,
):
    """Qwen3-TTS 训练一步前向（产出中间态 dict，供 Qwen3TTSLoss 算总 loss）。

    Args:
        model: `Qwen3TTSForConditionalGeneration`（底层，= `Qwen3TTSModel.model`），需含顶层
               `.speaker_encoder` + `.talker`（含 `.model.codec_embedding`、`.model.text_embedding`、
               `.text_projection`、`.codec_head`、`.code_predictor`、`.forward_sub_talker_finetune`）。
        batch: collate_qwen3_batch 输出转 torch 后的张量字典（input_ids[B,T,2]、codec_ids[B,T,16]、
               text_embedding_mask[B,T,1]、codec_embedding_mask[B,T,1]、codec_mask[B,T]、
               attention_mask[B,T]、codec_0_labels[B,T]、ref_mels[N,128]）。
        target_speaker_embedding: 首步算出的目标说话人 emb（官方写进 codec_embedding.weight[3000]；
               本项目走 clone 推理，可传 None，仅用于一致性记录）。

    Returns:
        dict:
          - 'talker_loss': 码本0 的 mean CE（model.talker.forward 的 outputs.loss，标准时间轴 shift 正确）。
          - 'sub_talker_logits': code_predictor 输出 [N, 15, V]（N=批次目标帧总数；15=码本1-15 头数）。
          - 'sub_talker_labels': [N, 15] = talker_codec_ids[:, 1:]（码本1-15 真值，已是 head 预测目标序列）。
          - 'target_speaker_embedding': 首步缓存的 speaker emb（供外部记录）。
        总 loss = talker_loss + 0.3 × code_predictor CE，由 Qwen3TTSLoss 从本 dict 算（不在编排里算）。
    """
    talker = model.talker
    # ---- ① speaker_encoder 前置（独立前向，detach 冻结）----
    speaker_embedding = model.speaker_encoder(batch["ref_mels"].to(model.dtype)).detach()
    if target_speaker_embedding is None:
        target_speaker_embedding = speaker_embedding

    # ---- ② 手动组装 inputs_embeds（不走 model(input_ids=...)）----
    input_text_ids = batch["input_ids"][..., 0]        # [B,T]
    input_codec_ids = batch["input_ids"][..., 1]      # [B,T]
    text_emb_mask = batch["text_embedding_mask"].to(model.dtype)
    codec_emb_mask = batch["codec_embedding_mask"].to(model.dtype)
    codec_ids = batch["codec_ids"]                    # [B,T,16]
    codec_mask = batch["codec_mask"]                  # [B,T]

    input_text_emb = talker.model.text_embedding(input_text_ids) * text_emb_mask
    input_codec_emb = talker.model.codec_embedding(input_codec_ids) * codec_emb_mask
    input_codec_emb[:, 6, :] = speaker_embedding      # 注入 pos6
    input_embeddings = input_text_emb + input_codec_emb

    # 叠加码本 1-15 的输入嵌入（code_predictor.get_input_embeddings() 返回 ModuleList）
    cp_input_embs = talker.code_predictor.get_input_embeddings()   # ModuleList[15]
    for i in range(1, talker.config.num_code_groups):
        e = cp_input_embs[i - 1](codec_ids[..., i]) * codec_mask.unsqueeze(-1).to(model.dtype)
        input_embeddings = input_embeddings + e

    # ---- ③ 第一段 forward：Talker 预测码本 0（inputs_embeds 与 labels 已 shift）----
    outputs = talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=batch["attention_mask"][:, :-1],
        labels=batch["codec_0_labels"][:, 1:],
        output_hidden_states=True,
    )
    talker_loss = outputs.loss                            # 码本0 mean shift+CE（HF loss_function，时间轴正确）

    # ---- ④ 第二段 forward：Code predictor 预测码本 1-15（向量化，一次喂所有目标帧）----
    # 注：forward_sub_talker_finetune 已向量化（输入 [N,16]+[N,H] → 内部 [N,16,H] 一次前向），
    #     非"逐帧循环"（docs/05 §3.2 / docs/06 §5.2 原述有误，W-4 修正）。
    hidden = outputs.hidden_states[0][-1]                 # [B, T-1, H]
    talker_hidden = hidden[codec_mask[:, :-1]]            # [N, H]，N=批次目标帧总数
    talker_codec_ids = codec_ids[codec_mask]             # [N, 16]
    sub_talker_logits, _sub_loss_ignored = talker.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden
    )
    # 忽略 _sub_loss_ignored（继承的 ForCausalLMLoss 对码本轴有 off-by-one，见 loss.py）；
    # Qwen3TTSLoss 用 sub_talker_logits + sub_talker_labels 自己算正确的 per-head CE。
    sub_talker_labels = talker_codec_ids[:, 1:]          # [N, 15] = 码本1-15 真值

    return {
        "talker_loss": talker_loss,
        "sub_talker_logits": sub_talker_logits,          # [N, 15, V]
        "sub_talker_labels": sub_talker_labels,           # [N, 15]
        "target_speaker_embedding": target_speaker_embedding,
    }


# --------------------------------------------------------------------
# Twinkle 集成选项（W-2 设计决策，待 W-5 NPU 验证后敲定）
# --------------------------------------------------------------------
# 选项 A（推荐）：自定义 TransformersModel 子类，override forward_backward/calculate_loss
#   - 编排：调 qwen3_tts_forward_step 拿中间态 dict，stash {'talker_loss','sub_talker_logits',
#     'sub_talker_labels'} 进 status.outputs（ModelOutput dict，自定义 key 运行时可塞）；
#     保留 clip_grad/梯度累积/metric 抽象 + 多 LoRA（adapter_name）透传。
#   - loss：set_loss(Qwen3TTSLoss)；calculate_loss 调 Qwen3TTSLoss(inputs, outputs) 从 stash 算总 loss。
#   - cookbook 仍调 model.forward_backward(inputs=batch)；编排全在子类内。
#   - 代价：要懂 twinkle.model.transformers.TransformersModel 的 forward/backward/calculate_loss
#     内部（transformers.py:851-869/757-805），override 时保留 clip_grad_and_step/梯度累积逻辑。
#
# 选项 B：自定义 task Patch（仿 patch/transformers_fused_ce.py 的 TransformersFusedCEPatch）
#   - 在 _resolve_task_context（transformers.py:85-103）注册新 task（如 'qwen3_tts'）；
#     patch 劫持 talker.forward 做 2 段 + speaker 前置 + stash 中间态。
#   - 优点：与 Twinkle task/Patch 范式一致；对 cookbook 透明（task='qwen3_tts'）。
#   - 代价：patch 劫持点要精准（需拿到 inputs_embeds 组装时机 + output_hidden_states）；多 LoRA 透传
#     需确认 patch 能访问 optimizer_group。
#
# 选项 C：绕过 forward_backward，cookbook 里直接调 qwen3_tts_forward_step + Qwen3TTSLoss 算 loss
#   + accelerator.backward（最接近官方 sft_12hz.py，但放弃 Twinkle 的 grad clip/lr step/metric 抽象，
#   最不"Twinkle"）。
#
# 倾向 A：编排复杂但可控，保留 Twinkle 训练循环抽象。W-5 去机房实测验证后定。
