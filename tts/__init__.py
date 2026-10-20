"""本项目 TTS 训练组件（开发期；成果稳定后回流 twinkle/src/twinkle/）。

结构（通用框架层 + 模型 cookbook）：
- `tts/common/`：模型无关的通用基类（PR 回流 twinkle 的目标）
    - `MultiCodebookPreprocessor`：manifest 行 → Trajectory（audio_codes 走 user_data）
    - `MultiCodebookLoss`：多码本 next-token CE 契约基类（W-4 填）
- `tts/qwen3/`：Qwen3-TTS cookbook（特有）
    - `data.py`：双通道 InputFeature 构建（移植官方 collate_fn，已单测）
    - `processor.py`：Qwen3TTSProcessor（用通用基类）
    - `template.py`：Qwen3TTSTemplate（override _encode_messages，W-3）
    - `forward.py`：qwen3_tts_forward_step（官方 5 步前向编排）
    - `loss.py`：Qwen3TTSLoss（2 段 CE，W-4）

架构见 `docs/05-W1-多码本组织调研.md`；设计见 `docs/06-W2-组件设计.md`。
"""

from .common import MultiCodebookLoss, MultiCodebookPreprocessor

__all__ = ["MultiCodebookPreprocessor", "MultiCodebookLoss"]
