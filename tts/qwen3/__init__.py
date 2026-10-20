"""Qwen3-TTS cookbook（特有部分）。

复用通用基类 `tts.common`，实现 Qwen3-TTS 特定的：
- 数据构建（双通道 InputFeature）：`data.py` 的 `build_qwen3_sample` + `collate_qwen3_batch`
- Processor：`Qwen3TTSProcessor`（用通用基类，Qwen3 默认字段名）
- Template：`Qwen3TTSTemplate`（override _encode_messages，W-3）
- 前向编排：`forward.py` 的 `qwen3_tts_forward_step`（官方 5 步）
- Loss：`Qwen3TTSLoss`（2 段 CE，W-4）

架构见 `docs/05-W1-多码本组织调研.md`；设计见 `docs/06-W2-组件设计.md`。
"""

from .data import Qwen3TTSDataConfig, build_qwen3_sample, collate_qwen3_batch
from .input_processor import Qwen3TTSInputProcessor
from .processor import Qwen3TTSProcessor

__all__ = [
    "Qwen3TTSDataConfig",
    "build_qwen3_sample",
    "collate_qwen3_batch",
    "Qwen3TTSProcessor",
    "Qwen3TTSInputProcessor",
]
