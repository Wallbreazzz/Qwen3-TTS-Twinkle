"""Qwen3-TTS Processor（cookbook）。

Qwen3 官方 manifest 字段名（audio/text/ref_audio/language）正好匹配通用基类默认值，
故 `Qwen3TTSProcessor` 直接继承 `MultiCodebookPreprocessor` 即可——仅作语义标记 +
将来若 Qwen3 字段名有偏差时在此覆盖。真正模型特定的重活在 `Qwen3TTSTemplate`
（构建双通道 InputFeature，见 template.py）。
"""

from tts.common import MultiCodebookPreprocessor


class Qwen3TTSProcessor(MultiCodebookPreprocessor):
    """manifest 行 → Trajectory（Qwen3-TTS 字段名与通用基类一致）。"""

    # Qwen3 官方 manifest（Qwen3-TTS/finetuning/prepare_data.py 输入）字段名：
    # {"audio":..., "text":..., "ref_audio":..., "audio_codes":[[...]], "language":...}
    # 与 MultiCodebookPreprocessor 默认完全一致，无需覆盖。
    pass
