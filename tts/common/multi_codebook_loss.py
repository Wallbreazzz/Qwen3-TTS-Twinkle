"""通用多码本 Loss 基类（模型无关，W-4 填实现）。

契约（子类按模型实现 `__call__`）：
- 多码本 next-token CE：每个码本一路 CE，按 `codebook_weights` 加权求和。
- `ar_codebook`：哪个码本由 AR 主模型预测（Qwen3=码本0 由 talker 算；其余由 depth
  transformer 算）——该路由决定是否走标准 shift+CE，残差码本走各自模型的前向。
- 设 `require_logps=False, require_logits=True`：跳过 Twinkle 单码本
  `selective_log_softmax`（`utils/torch_utils.py:62`），保留 `outputs['logits']`
  （`model/transformers/transformers.py:592/617`），由子类自算多码本 CE。

具体模型子类：
- Qwen3 2 段式（talker CE + 0.3×code_predictor CE）：见 `tts.qwen3.loss.Qwen3TTSLoss`
- MOSS 时间同步 / CosyVoice：各自 cookbook 子类（W-6）

⚠️ Qwen3 off-by-one（W-4 实测确认，见 `tts/qwen3/loss.py` + `tts/qwen3/tests/test_loss.py`）：
官方 code_predictor 的 forward_finetune 对 [N,15,V] logits 调继承的 `ForCausalLMLoss`，后者在
最后一轴做标准 LM shift → 对码本轴错位（head 预测码本 j+1 却对码本 j+2 算 loss，且 cb15 被末位
-100 pad 吞掉不算）。同位 per-head CE（无 shift）才是 depth-transformer next-token 的正确语义。
子类约定：code_predictor 段用 `shift_labels` 开关——False（默认）= 同位 per-head CE（修正），
True = 复现官方含 shift（W-5 NPU 实测对照用）。talker 段（标准时间轴 [B,T,V]）shift 正确，透传。
"""

from typing import Optional, Sequence

from twinkle.data_format.output import LossOutput
from twinkle.loss.base import Loss


class MultiCodebookLoss(Loss):
    """多码本 next-token CE 基类。

    Args:
        codebook_weights: 各码本 CE 权重，None=等权（Qwen3 的残差段权重 0.3 在子类里体现）。
        ar_codebook: AR 主码本 index（该码本由主模型 next-token 预测，走标准 shift+CE）。
    """

    require_logps = False
    require_logits = True

    def __init__(
        self,
        codebook_weights: Optional[Sequence[float]] = None,
        ar_codebook: int = 0,
    ):
        self.codebook_weights = codebook_weights
        self.ar_codebook = ar_codebook

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        raise NotImplementedError(
            "子类实现：按模型多码本组织算 CE。Qwen3 见 tts.qwen3.loss.Qwen3TTSLoss。"
        )
