"""Qwen3-TTS InputProcessor（cookbook，W-3）。

扩展 Twinkle `InputProcessor` 以支持 Qwen3-TTS 的双通道 + 多码本字段：

1. `padding_map`：注册 codec_ids/codec_0_labels/masks/ref_mels 的 pad 值。
2. `collate_fn` 覆写：2D 字段（input_ids[T,2]、codec_ids[T,16]、masks[T,1]）
   强制 **stack**（concat=False）→ [B,maxT,*]；ref_mels 沿时间 **cat**（镜像官方
   `torch.cat(ref_mels, dim=0)`，不 pad）。
   背景：基类 `_pad_sequence`→`pad_and_stack_tensors` 对 dim>=2 默认 concat=True
   （padding-free 拼接）→ 2D 字段会被错拼成 [ΣT,*] 而非 [B,maxT,*]；仅 routed_experts
   被显式 concat=False（`processor/base.py:776`），无类属性 hook 让 cookbook 追加，
   故此处接管 collate。
3. `to_transformers_dict` 覆写：把自定义字段加进白名单，避免被静默丢弃
   （基类用硬编码 `_keys`，未列入的字段直接 `continue` 跳过，见 `processor/base.py:663-680`）。

W-3 CPU 单测直接调 `collate_fn`/`to_transformers_dict`（不走 `prepare_inputs`，
避开 Platform 设备放置——CPU 机 `get_local_device()` 回退 GPU 返回 'cuda:0' 会失败；
真管线联调留 W-5）。
"""

from typing import Dict, List

import numpy as np
import torch

from twinkle.data_format import InputFeature
from twinkle.processor.base import InputProcessor
from twinkle.utils.torch_utils import pad_and_stack_tensors

# 序列字段：强制 stack（concat=False）→ [B, maxT, *]
_SEQ_STACK_FIELDS = (
    'input_ids', 'codec_ids', 'codec_0_labels',
    'text_embedding_mask', 'codec_embedding_mask', 'codec_mask',
    'attention_mask', 'position_ids',
)
# mask 字段 stack 后补 channel 维 → [B, maxT, 1]（对齐官方 collate_fn + forward 广播）
_MASK_CHANNEL_FIELDS = ('text_embedding_mask', 'codec_embedding_mask')

# to_transformers_dict 白名单追加（基类 `_keys` 之外的自定义字段）
_EXTRA_WHITELIST = (
    'codec_ids', 'codec_0_labels',
    'text_embedding_mask', 'codec_embedding_mask', 'codec_mask',
    'ref_mels',
)


def _to_tensor(v) -> torch.Tensor:
    """numpy/list → torch；bool → uint8（规避 F.pad 对 bool 的版本兼容问题；值仍是 0/1）。"""
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)
    elif isinstance(v, list):
        v = torch.tensor(v)
    if v.dtype == torch.bool:
        v = v.to(torch.uint8)
    return v


class Qwen3TTSInputProcessor(InputProcessor):
    """Qwen3-TTS 批组织：2D 强制 stack + ref_mels cat + 自定义字段白名单。"""

    padding_map = {
        **InputProcessor.padding_map,
        'codec_ids': 0,
        'codec_0_labels': -100,
        'text_embedding_mask': False,
        'codec_embedding_mask': False,
        'codec_mask': False,
        'ref_mels': 0.0,
    }

    def collate_fn(self, inputs, micro_batch_size=None, variable_seq_lengths=False, **kwargs):
        # 注：micro_batch_size 切分留 W-5（单卡 TTS 一次一个 macro batch 足够）。
        seq_out: Dict[str, torch.Tensor] = {}

        for key in _SEQ_STACK_FIELDS:
            vals: List[torch.Tensor] = []
            for inp in inputs:
                v = inp.get(key)
                if v is None:
                    continue
                v = _to_tensor(v)
                # 撤销 prepare_inputs 对 1D 字段的 unsqueeze（[1,T]→[T]）；
                # 2D 字段（[T,2]/[T,16]）shape[0]=T≠1 不受影响。
                if v.dim() == 2 and v.shape[0] == 1:
                    v = v.squeeze(0)
                vals.append(v)
            if not vals:
                continue
            pad = int(self.padding_map.get(key, 0))
            if len(vals) == 1:
                stacked = vals[0].unsqueeze(0)                # 保证有 batch 维
            else:
                stacked = pad_and_stack_tensors(vals, pad_value=pad, concat=False)
            if key in _MASK_CHANNEL_FIELDS:
                stacked = stacked.unsqueeze(-1)                # [B, maxT, 1]
            seq_out[key] = stacked

        # ref_mels：沿时间 cat（不 pad），镜像官方 torch.cat(ref_mels, dim=0)
        ref_mels: List[torch.Tensor] = []
        for inp in inputs:
            v = inp.get('ref_mels')
            if v is None:
                continue
            v = _to_tensor(v)
            ref_mels.append(v)
        if ref_mels:
            seq_out['ref_mels'] = torch.cat(ref_mels, dim=0)
        else:
            # 占位（W-5 的 mel 提取阶段会注入真实 ref_mels）。
            # 形状 [batch, mel_frames, 128]（3 维，speaker_encoder.transpose(1,2) 要 3 维），
            # 空占位用 (0, 0, 128)——W-5 实测确认 ref_mels 是 3 维非 2 维（见 docs/05 §4）。
            seq_out['ref_mels'] = torch.zeros((0, 0, 128), dtype=torch.float32)

        # length：标量元数据，取 batch 内最大
        lengths = [int(inp.get('length')) for inp in inputs if inp.get('length') is not None]
        if lengths:
            seq_out['length'] = max(lengths)

        return [InputFeature(**seq_out)]

    @staticmethod
    def to_transformers_dict(inputs, **kwargs):
        results = []
        # 基类白名单 + 本模型自定义字段（基类 _keys 见 processor/base.py:663-680）
        _keys = {
            'input_ids', 'inputs_embeds', 'attention_mask', 'position_ids', 'labels',
            'completion_mask', 'cu_seq_lens_q', 'cu_seq_lens_k', 'cu_seqlens_q',
            'cu_seqlens_kv', 'max_length_q', 'max_length_k', 'packed_seq_params',
            'routed_experts', 'mm_token_type_ids', 'second_per_grid_ts',
        } | set(InputProcessor.VLM_CONCAT_FIELDS) | set(_EXTRA_WHITELIST)

        for _input in inputs:
            output = {}
            for key in list(_input.keys()):
                if key not in _keys:
                    continue
                value = _input[key]
                if isinstance(value, torch.Tensor) or not isinstance(value, (list, np.ndarray)):
                    output[key] = value
                else:
                    output[key] = np.array(value)
            results.append(InputFeature(**output))
        return results
