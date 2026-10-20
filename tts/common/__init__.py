"""通用多码本 TTS 组件（框架层目标，成果稳定后回流 `twinkle/src/twinkle/`）。

这里放**模型无关**的抽象：
- `MultiCodebookPreprocessor`：manifest 行 → Trajectory（audio_codes 走 user_data）
- `MultiCodebookLoss`：多码本 next-token CE 的契约基类

具体模型（Qwen3 / MOSS / CosyVoice）在各自 cookbook 子目录里子类化实现。
maintainer 确认：audio_codes 放 `Trajectory.user_data`，不改 Message（见 docs/02 §4.3）。
"""

from .multi_codebook import MultiCodebookPreprocessor
from .multi_codebook_loss import MultiCodebookLoss

__all__ = ["MultiCodebookPreprocessor", "MultiCodebookLoss"]
