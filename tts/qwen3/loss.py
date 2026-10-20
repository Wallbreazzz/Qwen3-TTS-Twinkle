"""Qwen3-TTS Loss（cookbook，W-4 实现）。

Qwen3 是 2 段式：loss = talker CE(码本0) + 0.3 × code_predictor CE(码本1-15)。

- talker 段（码本0，[B,T,V] 标准时间轴 next-token）：编排从 model.talker.forward 拿 outputs.loss
  （HF 继承的 ForCausalLMLoss 对时间轴的 shift 是正确的），本类透传。
- code_predictor 段（码本1-15，[N,15,V] 沿码本轴 depth-transformer next-token）：本类显式算
  per-head CE。

⚠️ W-4 实测确认 off-by-one（见 docs/05 §5/§8 + tts/qwen3/tests/test_loss.py）：
  官方 forward_finetune（modeling_qwen3_tts.py:1242）对 [B,15,V] logits 调继承的
  self.loss_function（=ForCausalLMLoss），它在最后一轴做标准 LM shift：
    shift_labels = pad(labels,(0,1),-100)[...,1:]
  导致 head j（用 pos(j+1) 的 hidden 预测码本 j+1）被错配到码本 j+2 的 label，
  且 cb15 的预测被末位 -100 pad 吞掉（不算 loss），且 cb1 监督被首切丢失
  （labels[...,1:] 丢 labels[:,0]=cb1）——**双重偏移**：forward_sub_talker_finetune
  已把 labels 预处理成 codec_ids[:,1:]（预对齐，labels[:,j]=cb[j+1]，和 logits[:,j]
  天然对齐），ForCausalLMLoss 又 shift 一次。**这是 Qwen3-TTS 官方 bug**（会话9 Task agent
  读 modeling_qwen3_tts.py:1612-1633 源码确认），待本项目做完有 WER/SIM 证据时提 bugfix PR。
  同位 per-head CE（logits[:,j] ↔ labels[:,j]=码本 j+1，无 shift）才是 depth-transformer
  next-token 的正确语义——depth-transformer 的"next-token"已体现在"pos(j+1) 输入码本 j emb，
  head 读 pos(j+1) hidden 预测码本 j+1"里，labels 已是 head 预测目标序列，不需再 shift。

shift_labels 开关（W-5 NPU 实测定默认）：
  - False（默认，修正）：同位 per-head CE，不 shift。
  - True（复现官方）：复刻 ForCausalLMLoss 的 shift（含 off-by-one），供 W-5 对照。

集成（选项A 自定义 TransformersModel 包装器，W-5 落实）：编排 qwen3_tts_forward_step 产出
中间态 dict，包装器把 {'talker_loss','sub_talker_logits','sub_talker_labels'} stash 进
status.outputs；本类从 outputs 读这些算总 loss（total 不在编排里算，职责清晰）。
"""

import torch
import torch.nn.functional as F

from twinkle.data_format import LossOutput

from tts.common import MultiCodebookLoss


def _shift_ce_like_official(logits, labels, ignore_index=-100):
    """复刻 transformers ForCausalLMLoss 的 shift 语义（含 off-by-one，仅供 shift_labels=True 对照）。

    logits [N, C, V]，labels [N, C]。
    """
    from transformers.loss.loss_utils import ForCausalLMLoss

    return ForCausalLMLoss(
        logits=logits,
        labels=labels,
        vocab_size=logits.shape[-1],
        ignore_index=ignore_index,
    )


class Qwen3TTSLoss(MultiCodebookLoss):
    """Qwen3-TTS 2 段式 loss。

    Args:
        sub_loss_weight: code_predictor 段权重（官方 0.3）。
        shift_labels: code_predictor 段是否做标准 LM shift。
            False（默认）= 修正 off-by-one，同位 per-head CE（无 shift）；
            True = 复现官方 ForCausalLMLoss 含 shift（W-5 实测对照用）。
        ignore_index: 忽略的 label 值（非音频帧 / pad）。
    """

    def __init__(
        self,
        sub_loss_weight: float = 0.3,
        shift_labels: bool = False,
        ignore_index: int = -100,
        ar_codebook: int = 0,
    ):
        super().__init__(ar_codebook=ar_codebook)
        self.sub_loss_weight = sub_loss_weight
        self.shift_labels = shift_labels
        self.ignore_index = ignore_index

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        # talker 段（码本0）：透传编排算好的 mean CE（标准时间轴 shift 正确）
        talker_loss = outputs["talker_loss"]
        # code_predictor 段（码本1-15）
        sub_logits = outputs["sub_talker_logits"]   # [N, 15, V]
        sub_labels = outputs["sub_talker_labels"]   # [N, 15]

        if self.shift_labels:
            sub_loss = _shift_ce_like_official(sub_logits, sub_labels, self.ignore_index)
        else:
            V = sub_logits.shape[-1]
            # 同位 per-head CE（无 shift）：logits[:,j] 预测码本 j+1 ↔ labels[:,j]=码本 j+1
            sub_loss = F.cross_entropy(
                sub_logits.reshape(-1, V),
                sub_labels.reshape(-1),
                ignore_index=self.ignore_index,
                reduction="mean",
            )

        total = talker_loss + self.sub_loss_weight * sub_loss
        # 两段均 per-token mean（镜像官方 sft_12hz.py：outputs.loss 与 sub_loss 都是 mean）。
        # num_tokens=0 走 Twinkle PER-TOKEN-MEAN 归一（grad_accum 路径），
        # 与 accelerate.backward(loss)+gradient_accumulation_steps 一致。
        return LossOutput(loss=total, num_tokens=0)
