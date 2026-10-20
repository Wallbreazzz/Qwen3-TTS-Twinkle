"""Qwen3TTSTemplate 的 CPU 单测（W-3）。

两部分：
1. fake-tokenizer 单测：注入确定性伪 tokenizer，验证 `_encode_messages` 接线
   （encode→[:-TEXT_IDS_TRIM]→build_qwen3_sample→InputFeature）+ 后管线覆写
   （`_add_attention_fields` 对 2D input_ids 产出 [T] attention_mask、`_roll_labels` no-op 不崩）。
2. real-tokenizer 集成测：下载真实 Qwen3-TTS 文本 tokenizer，验证 `[:-5]` 切片边界
   在 `<|im_end|>` 前（若计数不符会报具体尾部 token）。下载失败则 SKIP。

运行：python tts/qwen3/tests/test_template.py
（或 PYTHONPATH=twinkle/src python tts/qwen3/tests/test_template.py）
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..'))                       # 项目根 → import tts
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'twinkle', 'src'))     # twinkle

# Windows GBK 控制台无法打印 BPE token 字符（Ġ/Ċ 等），改 UTF-8 输出避免 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from twinkle.data_format import Message, Trajectory, pack_value

from tts.qwen3.data import Qwen3TTSDataConfig
from tts.qwen3.template import TEXT_IDS_TRIM, Qwen3TTSTemplate, build_assistant_text


def _fake_cfg():
    return Qwen3TTSDataConfig(
        tts_pad_token_id=0, tts_bos_token_id=1, tts_eos_token_id=2,
        codec_pad_id=10, codec_bos_id=11, codec_eos_token_id=12,
        codec_nothink_id=13, codec_think_bos_id=14, codec_think_eos_id=15,
        num_code_groups=16,
    )


class _FakeTokenizer:
    """确定性伪 tokenizer：encode 忽略文本，返回固定 10 个 id（[:-5] 后 5 个，>=4 满足布局）。"""

    def encode(self, text, **kwargs):
        return list(range(1000, 1010))


def _new_template():
    # 绕过 Template.__init__（会下载 tokenizer/config），手动注入测试所需属性。
    # tokenizer 是只读 property（返回 self.processor），故设 processor。
    t = Qwen3TTSTemplate.__new__(Qwen3TTSTemplate)
    t.data_config = _fake_cfg()
    t.processor = _FakeTokenizer()
    t.max_length = None
    return t


def _fake_trajectory(text='你好世界', ref='wavs/ref.wav', C=4):
    audio_codes = np.zeros((C, 16), dtype=np.int64)
    for i in range(C):
        for j in range(16):
            audio_codes[i, j] = 1000 + i * 16 + j
    audio_codes[:, 0] = np.arange(200, 200 + C, dtype=np.int64)
    return Trajectory(
        messages=[Message(role='user', content=text)],
        user_data=[
            ('audio_codes', pack_value(audio_codes.tolist())),
            ('ref_audio', pack_value(ref)),
            ('language', pack_value('Chinese')),
        ],
    )


def test_encode_messages_wiring():
    t = _new_template()
    traj = _fake_trajectory()
    feat = t._encode_messages(traj)

    # text_ids = [1000..1009][:-5] = 5 个；L=5, C=4 → T=8+5+4=17
    T = 17
    assert feat['input_ids'].shape == (T, 2), feat['input_ids'].shape
    assert feat['codec_ids'].shape == (T, 16)
    assert feat['codec_0_labels'].shape == (T,)
    assert int(feat['length']) == T
    # 文本通道头部 = text_ids[:3]
    assert feat['input_ids'][:3, 0].tolist() == [1000, 1001, 1002]
    # ref_audio 路径已存
    assert feat['ref_audio'] == 'wavs/ref.wav'
    # attention_mask 是 [T]（1D），不是 [T,2]
    assert feat['attention_mask'].shape == (T,), feat['attention_mask'].shape
    # 不设标准 labels 键（用 codec_0_labels，shift 留给 forward 编排）
    assert 'labels' not in feat
    print('OK _encode_messages wiring, T=', T)


def test_post_pipeline_attention_mask_not_clobbered():
    # 后管线：_check_max_length(max=None,no-op) → _add_attention_fields → _roll_labels
    t = _new_template()
    feat = t._encode_messages(_fake_trajectory())
    feats = t._invoke_post_pipeline([feat])
    f = feats[0]
    T = 17
    # _add_attention_fields 对 2D input_ids 不应把 attention_mask 覆盖成 [T,2]
    assert f['attention_mask'].shape == (T,), f['attention_mask'].shape
    # position_ids / length 已补
    assert f['position_ids'].shape == (T,)
    assert f['position_ids'].tolist() == list(range(T))
    assert int(f['length']) == T
    # _roll_labels no-op：不应新增 labels 键，也不应崩
    assert 'labels' not in f
    print('OK post-pipeline: attention_mask=[T], position_ids/length set, no labels')


def test_roll_labels_noop_without_labels():
    t = _new_template()
    feat = t._encode_messages(_fake_trajectory())
    # 直接调 _roll_labels：无 labels 键不应崩，且不引入 labels
    out = t._roll_labels(dict(feat))[0]
    assert 'labels' not in out
    print('OK _roll_labels no-op (no KeyError without labels)')


def test_add_attention_fields_2d_input_ids():
    t = _new_template()
    feat = t._encode_messages(_fake_trajectory())
    # 单独验证 2D 路径：attention_mask 保持 [T]
    out = t._add_attention_fields(dict(feat))[0]
    assert out['attention_mask'].shape == (17,)
    assert out['attention_mask'].dtype.kind in ('i', 'u')  # int 类
    print('OK _add_attention_fields 2D input_ids → attention_mask [T]')


def test_real_tokenizer_trim():
    """真实 Qwen3-TTS 文本 tokenizer 联调：验证 [:-5] 切片边界在 <|im_end|> 前。"""
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # noqa
        print(f'SKIP test_real_tokenizer_trim: transformers 不可用 ({e})')
        return
    model_id = os.environ.get('QWEN3_TTS_MODEL_ID', 'Qwen/Qwen3-TTS-12Hz-1.7B-Base')
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:  # noqa
        print(f'SKIP test_real_tokenizer_trim: 无法下载 tokenizer {model_id!r} ({e})')
        return

    text = '今天天气真好。'
    full = tok.encode(build_assistant_text(text))
    trimmed = full[:-TEXT_IDS_TRIM]
    tail = full[-TEXT_IDS_TRIM:]
    tail_tokens = tok.convert_ids_to_tokens(tail)

    end_id = tok.convert_tokens_to_ids('<|im_end|>')
    start_id = tok.convert_tokens_to_ids('<|im_start|>')
    print(f'full len={len(full)}, trimmed len={len(trimmed)}, tail tokens={tail_tokens}')

    # 切片边界应在 <|im_end|> 前（tail[0] 应是 <|im_end|>）——若计数不符此处会报错，
    # 提示把 TEXT_IDS_TRIM 改成实际值。
    assert tail[0] == end_id, (
        f'[:-{TEXT_IDS_TRIM}] 边界应在 <|im_end|> 前，但 tail[0]={tail_tokens[0]!r}；'
        f'完整 tail={tail_tokens}（可能需调整 TEXT_IDS_TRIM）')
    # 尾部应含 <|im_start|>（重开 assistant 轮）
    assert start_id in tail, f'尾部应含 <|im_start|>，实得 {tail_tokens}'
    # trim 后仍以 <|im_start|> 开头（assistant 前缀）
    assert start_id in trimmed[:3], (
        f'trim 后应以 <|im_start|> 开头，实得 {tok.convert_ids_to_tokens(trimmed[:3])}')
    print('OK real tokenizer trim（边界在 <|im_end|> 前，保留 assistant 前缀）')


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for tn in tests:
        print(f'\n--- {tn.__name__} ---')
        tn()
    print(f'\n全部通过：{len(tests)} 个测试（其中 real-tokenizer 集成测可能 SKIP）')


if __name__ == '__main__':
    _run_all()
