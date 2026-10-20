"""build_qwen3_sample / collate_qwen3_batch 的 CPU 单测（W-3）。

用伪 config + 伪输入验证双通道布局与官方 `Qwen3-TTS/finetuning/dataset.py` collate_fn 一致
（不变量断言，不依赖真实模型/tokenizer）。详见 `docs/05` 第 4 节。

运行：python tts/qwen3/tests/test_data.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np

from tts.qwen3.data import Qwen3TTSDataConfig, build_qwen3_sample, collate_qwen3_batch


def _fake_cfg():
    return Qwen3TTSDataConfig(
        tts_pad_token_id=0, tts_bos_token_id=1, tts_eos_token_id=2,
        codec_pad_id=10, codec_bos_id=11, codec_eos_token_id=12,
        codec_nothink_id=13, codec_think_bos_id=14, codec_think_eos_id=15,
        num_code_groups=16,
    )


def _fake_sample(L=6, C=4):
    text_ids = np.arange(100, 100 + L, dtype=np.int64)          # [100..105]
    audio_codes = np.zeros((C, 16), dtype=np.int64)
    for i in range(C):
        for j in range(16):
            audio_codes[i, j] = 1000 + i * 16 + j               # 可区分的码本值
    audio_codes[:, 0] = np.arange(200, 200 + C, dtype=np.int64)  # 码本0 覆写为 [200,201,202,203]
    return text_ids, audio_codes


def test_sample_shapes():
    cfg = _fake_cfg()
    text_ids, audio_codes = _fake_sample(L=6, C=4)
    s = build_qwen3_sample(text_ids, audio_codes, cfg)
    T = 8 + 6 + 4  # =18
    assert s["input_ids"].shape == (T, 2), s["input_ids"].shape
    assert s["codec_ids"].shape == (T, 16)
    assert s["codec_0_labels"].shape == (T,)
    for k in ("text_embedding_mask", "codec_embedding_mask", "codec_mask", "attention_mask"):
        assert s[k].shape == (T,), (k, s[k].shape)
    assert int(s["length"]) == T
    print("OK shapes T=", T)


def test_sample_layout_text_channel():
    cfg = _fake_cfg()
    text_ids, audio_codes = _fake_sample(L=6, C=4)
    s = build_qwen3_sample(text_ids, audio_codes, cfg)
    ch0 = s["input_ids"][:, 0]
    # [0:3]=text[:3]
    assert list(ch0[0:3]) == [100, 101, 102]
    # [3:7]=tts_pad(0)
    assert list(ch0[3:7]) == [0, 0, 0, 0]
    # [7]=tts_bos(1)
    assert ch0[7] == 1
    # [8:11]=text[3:]=[103,104,105]   (8 .. 8+6-3=11)
    assert list(ch0[8:11]) == [103, 104, 105]
    # [11]=tts_eos(2)
    assert ch0[11] == 2
    # [12:18]=tts_pad(0)
    assert list(ch0[12:18]) == [0, 0, 0, 0, 0, 0]
    print("OK text channel layout")


def test_sample_layout_codec_channel():
    cfg = _fake_cfg()
    text_ids, audio_codes = _fake_sample(L=6, C=4)
    s = build_qwen3_sample(text_ids, audio_codes, cfg)
    ch1 = s["input_ids"][:, 1]
    # [0:3]=0 (default)
    assert list(ch1[0:3]) == [0, 0, 0]
    # [3:8]=[nothink,think_bos,think_eos,0,pad]=[13,14,15,0,10]
    assert list(ch1[3:8]) == [13, 14, 15, 0, 10]
    # [8:11]=codec_pad(10)  (8..8+6-3=11)
    assert list(ch1[8:11]) == [10, 10, 10]
    # [11]=codec_pad(10)
    assert ch1[11] == 10
    # [12]=codec_bos(11)
    assert ch1[12] == 11
    # [13:17]=audio_codec_0=[200,201,202,203]
    assert list(ch1[13:17]) == [200, 201, 202, 203]
    # [17]=codec_eos(12)
    assert ch1[17] == 12
    print("OK codec channel layout")


def test_sample_codec_0_labels():
    cfg = _fake_cfg()
    text_ids, audio_codes = _fake_sample(L=6, C=4)
    s = build_qwen3_sample(text_ids, audio_codes, cfg)
    lab = s["codec_0_labels"]
    # 非目标帧 -100
    assert list(lab[0:13]) == [-100] * 13
    # [13:17]=audio_codec_0
    assert list(lab[13:17]) == [200, 201, 202, 203]
    # [17]=codec_eos(12)
    assert lab[17] == 12
    print("OK codec_0_labels")


def test_sample_codec_ids_and_mask():
    cfg = _fake_cfg()
    text_ids, audio_codes = _fake_sample(L=6, C=4)
    s = build_qwen3_sample(text_ids, audio_codes, cfg)
    # codec_ids 仅 [13:17] 有值
    assert s["codec_ids"][13:17].tolist() == audio_codes.tolist()
    # 其余全 0
    assert s["codec_ids"][:13].sum() == 0
    assert s["codec_ids"][17:].sum() == 0
    # codec_mask 仅 [13:17]=True（C=4 个，不含 eos pos17）
    assert list(s["codec_mask"][0:13]) == [False] * 13
    assert list(s["codec_mask"][13:17]) == [True, True, True, True]
    assert s["codec_mask"][17] == False
    # attention_mask [:18]=True
    assert s["attention_mask"].all()
    # text_embedding_mask [:18]=True
    assert s["text_embedding_mask"].all()
    # codec_embedding_mask [3:18]=True except [6]=False（pos6 在 [3:] 内 → 14 个 True）
    assert not s["codec_embedding_mask"][6]
    assert s["codec_embedding_mask"][3:].sum() == 14
    assert s["codec_embedding_mask"][:3].sum() == 0
    print("OK codec_ids / masks")


def test_collate_batch_padding():
    cfg = _fake_cfg()
    t0, a0 = _fake_sample(L=6, C=4)   # T=18
    t1, a1 = _fake_sample(L=5, C=3)   # T=16
    s0 = build_qwen3_sample(t0, a0, cfg)
    s1 = build_qwen3_sample(t1, a1, cfg)
    batch = collate_qwen3_batch([s0, s1], cfg)
    # batch shape
    assert batch["input_ids"].shape == (2, 18, 2)
    assert batch["codec_ids"].shape == (2, 18, 16)
    assert batch["codec_0_labels"].shape == (2, 18)
    assert batch["codec_mask"].shape == (2, 18)
    # 样本0 未被 pad（前18全有效）
    assert batch["attention_mask"][0].all()
    # 样本1 仅前16有效，后2为 padding
    assert batch["attention_mask"][1, :16].all()
    assert batch["attention_mask"][1, 16:].sum() == 0
    # 样本1 padding 段 codec_0_labels=-100
    assert (batch["codec_0_labels"][1, 16:] == -100).all()
    # 样本1 padding 段 input_ids=0
    assert (batch["input_ids"][1, 16:] == 0).all()
    # 样本1 padding 段 codec_mask=False
    assert not batch["codec_mask"][1, 16:].any()
    # mask 通道维 [b,t,1]
    assert batch["text_embedding_mask"].shape == (2, 18, 1)
    assert batch["codec_embedding_mask"].shape == (2, 18, 1)
    print("OK collate padding")


def test_collate_ref_mels_cat():
    cfg = _fake_cfg()
    t, a = _fake_sample(L=6, C=4)
    s = build_qwen3_sample(t, a, cfg)
    mel0 = np.zeros((10, 128), dtype=np.float32)
    mel1 = np.zeros((7, 128), dtype=np.float32)
    batch = collate_qwen3_batch([s, s], cfg, ref_mels=[mel0, mel1])
    # 沿时间 cat: 10+7=17
    assert batch["ref_mels"].shape == (17, 128)
    print("OK ref_mels cat")


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()
    print(f"\n全部通过：{len(tests)} 个测试")


if __name__ == "__main__":
    _run_all()
