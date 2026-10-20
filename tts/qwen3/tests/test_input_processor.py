"""Qwen3TTSInputProcessor 的 CPU 单测（W-3）。

验证（直接调 collate_fn / to_transformers_dict，不走 prepare_inputs，避开 Platform）：
- batch 形状：2D 字段 stack 成 [B, maxT, *]（非 padding-free concat 的 [ΣT, *]）；
- padding 值：短样本 padding 段 codec_0_labels=-100 / input_ids=0 / masks=0 / attention_mask=0；
- ref_mels 沿时间 cat（[Σmel, 128]）；
- 与 numpy 版 collate_qwen3_batch 逐字段数值一致（交叉核对）；
- to_transformers_dict 白名单：自定义字段保留、非白名单字段丢弃。

运行：python tts/qwen3/tests/test_input_processor.py
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'twinkle', 'src'))

from tts.qwen3.data import Qwen3TTSDataConfig, build_qwen3_sample, collate_qwen3_batch
from tts.qwen3.input_processor import Qwen3TTSInputProcessor


def _fake_cfg():
    return Qwen3TTSDataConfig(
        tts_pad_token_id=0, tts_bos_token_id=1, tts_eos_token_id=2,
        codec_pad_id=10, codec_bos_id=11, codec_eos_token_id=12,
        codec_nothink_id=13, codec_think_bos_id=14, codec_think_eos_id=15,
        num_code_groups=16,
    )


def _sample(L, C):
    text_ids = np.arange(100, 100 + L, dtype=np.int64)
    audio_codes = np.zeros((C, 16), dtype=np.int64)
    for i in range(C):
        for j in range(16):
            audio_codes[i, j] = 1000 + i * 16 + j
    audio_codes[:, 0] = np.arange(200, 200 + C, dtype=np.int64)
    return text_ids, audio_codes


def _feats(cfg):
    # s0: L=6,C=4 → T0=18；s1: L=5,C=3 → T1=16；maxT=18
    t0, a0 = _sample(6, 4)
    t1, a1 = _sample(5, 3)
    s0 = build_qwen3_sample(t0, a0, cfg)
    s1 = build_qwen3_sample(t1, a1, cfg)
    mel0 = np.zeros((10, 128), dtype=np.float32)
    mel1 = np.zeros((7, 128), dtype=np.float32)
    f0 = dict(s0); f0['ref_mels'] = mel0
    f1 = dict(s1); f1['ref_mels'] = mel1
    return [f0, f1], (s0, s1), (mel0, mel1)


def test_batch_shapes_2d_stack():
    cfg = _fake_cfg()
    proc = Qwen3TTSInputProcessor()
    feats, _, _ = _feats(cfg)
    b = proc.collate_fn(feats)[0]
    # 2D 字段 stack（dim0=B=2），不是 concat（ΣT=34）
    assert b['input_ids'].shape == (2, 18, 2), b['input_ids'].shape
    assert b['input_ids'].shape[0] == 2  # stack 不是 concat
    assert b['codec_ids'].shape == (2, 18, 16), b['codec_ids'].shape
    assert b['codec_ids'].shape[0] == 2
    # 1D 字段 stack → [B, maxT]
    assert b['codec_0_labels'].shape == (2, 18)
    assert b['codec_mask'].shape == (2, 18)
    assert b['attention_mask'].shape == (2, 18)
    # mask 补 channel 维 → [B, maxT, 1]
    assert b['text_embedding_mask'].shape == (2, 18, 1), b['text_embedding_mask'].shape
    assert b['codec_embedding_mask'].shape == (2, 18, 1)
    # ref_mels 沿时间 cat：10+7=17
    assert b['ref_mels'].shape == (17, 128), b['ref_mels'].shape
    assert int(b['length']) == 18
    print('OK batch shapes：2D stack [B,maxT,*]，ref_mels cat [Σmel,128]')


def test_padding_values():
    cfg = _fake_cfg()
    proc = Qwen3TTSInputProcessor()
    feats, _, _ = _feats(cfg)
    b = proc.collate_fn(feats)[0]
    # 样本1 T1=16，padding 到 18，后 2 位为 padding
    assert (b['codec_0_labels'][1, 16:] == -100).all()
    assert (b['input_ids'][1, 16:] == 0).all()
    assert (b['codec_ids'][1, 16:] == 0).all()
    assert (b['codec_mask'][1, 16:] == 0).all()
    assert (b['attention_mask'][1, 16:] == 0).all()
    assert (b['text_embedding_mask'][1, 16:, 0] == 0).all()
    # 样本0 未 padding（前 18 全有效）
    assert b['attention_mask'][0].all()
    print('OK padding values：codec_0_labels=-100 / input_ids=0 / masks=0 / attn=0')


def test_cross_check_with_numpy_collate():
    cfg = _fake_cfg()
    proc = Qwen3TTSInputProcessor()
    feats, (s0, s1), (mel0, mel1) = _feats(cfg)
    my = proc.collate_fn(feats)[0]
    npb = collate_qwen3_batch([s0, s1], cfg, ref_mels=[mel0, mel1])
    for k in ('input_ids', 'codec_ids', 'codec_0_labels', 'codec_mask',
             'attention_mask', 'text_embedding_mask', 'codec_embedding_mask', 'ref_mels'):
        mine = my[k].numpy()
        theirs = npb[k]
        assert np.array_equal(mine, theirs), f'{k}: 与 collate_qwen3_batch 不一致\nmine={mine}\ntheirs={theirs}'
    print('OK 与 collate_qwen3_batch 逐字段数值一致')


def test_whitelist_keeps_custom_drops_unknown():
    cfg = _fake_cfg()
    proc = Qwen3TTSInputProcessor()
    feats, _, _ = _feats(cfg)
    batch = proc.collate_fn(feats)[0]
    # 注入一个非白名单字段（模拟 trajectory 残留的 messages）
    with_extra = dict(batch)
    with_extra['messages'] = [{'role': 'user', 'content': 'x'}]
    out = Qwen3TTSInputProcessor.to_transformers_dict([with_extra])[0]
    for k in ('codec_ids', 'codec_0_labels', 'ref_mels',
              'text_embedding_mask', 'codec_embedding_mask', 'codec_mask'):
        assert k in out, f'{k} 被白名单静默丢弃'
    assert 'messages' not in out, '非白名单字段 messages 应被丢弃'
    # ref_mels 保持 torch 张量（未被转成 list/np）
    assert isinstance(out['ref_mels'], torch.Tensor)
    print('OK 白名单：自定义字段保留、非白名单丢弃、ref_mels 保张量')


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for tn in tests:
        print(f'\n--- {tn.__name__} ---')
        tn()
    print(f'\n全部通过：{len(tests)} 个测试')


if __name__ == '__main__':
    _run_all()
