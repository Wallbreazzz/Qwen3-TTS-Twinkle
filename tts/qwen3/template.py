"""Qwen3-TTS Template（cookbook，W-3 实现）。

override `Template._encode_messages`：从 `Trajectory.user_data` 取 audio_codes/text，
用文本 tokenizer 编码 `build_assistant_text(text)` → text_ids（去尾部 `TEXT_IDS_TRIM`
个闭合/重开 token），调 `build_qwen3_sample(text_ids, audio_codes, cfg)` 产出双通道
InputFeature（input_ids[T,2]/codec_ids[T,16]/codec_0_labels[T]/4 mask）。

后管线两处冲突，分别覆写（保留管线、贴框架；用户选定方案 B）：
- `_add_attention_fields`：基类对 `input_ids` 用 `np.ones_like`→ 对 2D `[T,2]` 会得
  `[T,2]`（错；forward 要 `[T]`）。1D（标准文本）走基类；2D（双通道）这里产出 `[T]`
  的 attention_mask（沿用 build_qwen3_sample 已有的全 1 mask，缺失/形状不对则补）+ position_ids + length。
- `_roll_labels`：基类无条件滚 `labels` 键，但 Qwen3-TTS 不用标准 `labels`（用
  `codec_0_labels`，shift 由 forward 编排 `[:, 1:]` 做，镜像官方 `sft_12hz.py`）；
  覆写为 no-op，避免 KeyError 与重复 shift。2D 障碍（把 `[T,16]` 当 label 滚错轴）
  也不会触发——`codec_ids` 是输入非 label，不进 `labels` 键。

ref_audio 路径存进 per-sample feature（mel 提取留 W-5；该字段不进白名单，会被
`to_transformers_dict` 丢弃，仅上游 mel 提取阶段用）。`max_length` 建议设 None
或大值（截断会破坏音频对齐，用默认 'raise' 策略由超长直接报错）。

架构见 `docs/05-W1-多码本组织调研.md` 第 4/6 节；设计见 `docs/06-W2-组件设计.md` 第 2 节。
"""

import numpy as np

from twinkle.data_format import Trajectory, user_data_get
from twinkle.data_format.input_feature import InputFeature
from twinkle.template.base import Template

from .data import Qwen3TTSDataConfig, build_assistant_text, build_qwen3_sample

# 官方 `_build_assistant_text` 编码后去尾部的 token 数（去掉
# `<|im_end|>\n<|im_start|>assistant\n` 这段闭合/重开标记）。经验值，真实 Qwen3 文本
# tokenizer 联调验证（见 tests/test_template.py 集成测；若计数不符该测会报具体尾部 token）。
TEXT_IDS_TRIM = 5


class Qwen3TTSTemplate(Template):
    """Qwen3-TTS 双通道 encode（W-3 实现）。"""

    def __init__(self, data_config: Qwen3TTSDataConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_config = data_config

    def _encode_messages(self, trajectory: Trajectory, add_generation_prompt: bool = False, **kwargs) -> InputFeature:
        # 训练路径：add_generation_prompt 忽略（推理编码留后续）。
        # 1. 从 user_data 取字段
        text = trajectory['messages'][0]['content']
        audio_codes = user_data_get(trajectory.get('user_data'), 'audio_codes')
        ref_audio = user_data_get(trajectory.get('user_data'), 'ref_audio')

        # 2. 文本 tokenizer 编码：build_assistant_text 把目标文本包成 assistant 轮（ChatML），
        #    encode 后去尾部 TEXT_IDS_TRIM 个闭合/重开 token，留下 assistant 前缀 + 目标文本。
        text_ids = self.tokenizer.encode(build_assistant_text(text))
        text_ids = np.array(text_ids, dtype=np.int64)[:-TEXT_IDS_TRIM]

        # 3. 组双通道 InputFeature（移植官方 collate_fn 的单样本布局）
        sample = build_qwen3_sample(text_ids, audio_codes, self.data_config)

        # 4. ref_audio 路径留存（mel 提取留 W-5）
        sample['ref_audio'] = ref_audio

        trajectory.update(sample)
        return trajectory

    def _add_attention_fields(self, input_feature: InputFeature) -> list:
        # 基类 `np.ones_like(input_ids)` 对 2D `[T,2]` 产出 `[T,2]`（错；forward 要 `[T]`）。
        # 1D（标准文本）走基类；2D（双通道）这里产出 `[T]` 的 attention_mask。
        input_ids = input_feature.get('input_ids')
        if input_ids is None:
            return [input_feature]
        if not isinstance(input_ids, np.ndarray) or input_ids.ndim <= 1:
            return super()._add_attention_fields(input_feature)
        T = int(input_ids.shape[0])
        am = input_feature.get('attention_mask')
        if not isinstance(am, np.ndarray) or am.ndim != 1 or am.shape[0] != T:
            # build_qwen3_sample 已给全 1 的 [T] mask；缺失/形状不对则补
            input_feature['attention_mask'] = np.ones(T, dtype=np.int64)
        input_feature['position_ids'] = np.arange(T)
        input_feature['length'] = T
        return [input_feature]

    def _roll_labels(self, input_feature: InputFeature) -> list:
        # Qwen3-TTS 不用标准 `labels` 键（用 `codec_0_labels`）；shift 由 forward 编排
        # 做（`codec_0_labels[:, 1:]`，镜像官方 sft_12hz.py），故此步 no-op，避免 KeyError
        # 与重复 shift。`codec_ids[T,16]` 是输入非 label，不进 `labels` 键，2D 滚轴障碍不触发。
        return [input_feature]
