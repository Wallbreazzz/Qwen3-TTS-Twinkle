# -*- coding: utf-8 -*-
"""客户录制数据集后处理流水线（CPU 本地版，docs/09 §6 十步中除 audio_codes 编码外的全部）。

输入:
  客户录制数据集/                      162 个 48k/16bit/mono WAV，命名 `X - Y.wav`(X=遍数,Y=语料号)
  客户录制数据集/数据集应该的对应文本.txt   原稿文本(编号\t文本, 含跨行引号条目)
  客户录制数据集/数据集质量分析报告.json   会话11 体检报告(复用 snr_db 等, 不重算)

输出(全部写到 客户录制数据集_processed/, 原始数据一律不动):
  excluded/          必删 3 条的副本 + 原因清单
  wavs/              处理成品: 24kHz/mono/16bit, -23LUFS, 首尾静音 300ms, 超30s已VAD切分
  audio_log.jsonl    每个输出文件的处理流水账(来源/时长/增益/VAD段数/标记)
  corpus_texts.json  解析后的语料号→原稿文本
  asr_report.jsonl   ASR 校验: 每条音频转写 vs 原稿, 相似度, 低分标红待人工
  tn_review.csv      文本归一化(TN)人工复核清单(数字/工号/电话/占位符)
  train.jsonl        manifest 草稿(对齐 prepare_aishell3.py 字段, audio_codes/ref_mel 留容器内填)
  dev.jsonl          同上, val 侧(与 train 文本不重叠, 含近似重复文本分组防泄漏)
  听辨清单.html      SNR 边界值文件逐条听辨页(勾选排除, 可导出 JSON)
  处理报告.md        汇总统计

阶段(可单独重跑):
  --stage text     解析原稿文本 + TN 标记
  --stage audio    删异常/VAD切分/静音修整/响度归一/下采样 → wavs/
  --stage asr      faster-whisper small 校验(含切分段的文本对齐 DP)
  --stage finalize train/dev 拆分 + manifest 草稿 + 听辨清单 + 报告
  --stage all      顺序全跑

用法:
  python scripts/postprocess_customer_dataset.py --stage all
依赖: numpy soundfile soxr librosa pyloudnorm silero-vad faster-whisper(仅 asr 阶段)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / '客户录制数据集'
OUT_DIR = ROOT / '客户录制数据集_processed'
TEXT_TXT = DATA_DIR / '数据集应该的对应文本.txt'
QUALITY_JSON = DATA_DIR / '数据集质量分析报告.json'

SR_IN = 48000          # 源采样率
SR_OUT = 24000         # Qwen3-TTS 硬要求
TARGET_LUFS = -23.0    # docs/09 §6 步骤5
EDGE_SILENCE_MS = 300  # docs/03: 成品首尾静音 300ms
MAX_EDGE_SILENCE_MS = 800  # 超过则裁到 EDGE_SILENCE_MS(省 codec token)
SILENCE_DB = -40.0     # 与 analyze_dataset_quality.py 一致的静音判定阈值(dBFS)
LONG_FILE_S = 30.0     # 超过则 VAD 切分(docs/09 §6 步骤2)
SEG_TARGET_S = 15.0    # 切分目标段长上限(不切语音内部, 允许略超)
SEG_HARD_MAX_S = 30.0  # 合并碎段后的硬上限
SEG_MIN_S = 1.5        # 短于此的碎段: 能并则并, 不能并则丢
PEAK_CEIL_DBFS = -3.0  # 归一后峰值上限(防削波)
SNR_BORDERLINE_DB = 35.0  # docs/09: SNR<35 为边界值, 待人工听辨

# docs/09 §4: 必删 3 条 (take, corpus) → 原因
EXCLUDE = {
    (2, 82): '全静音损坏(peak=-240dBFS)',
    (1, 1): '过短2.5s + SNR低31.8dB + 噪声高-56.2dBFS',
    (2, 1): '过短2.5s + SNR低27.6dB + 噪声高-52.6dBFS + 首尾静音不足',
}

FNAME_RE = re.compile(r'^(\d+)\s*-\s*(\d+)\.wav$')


def parse_fname(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------- 文本解析(TN)

def parse_corpus_texts(txt_path: Path) -> dict[int, str]:
    """解析 `编号\\t文本` 格式; 处理跨行引号条目(23/24/60等)与 49/50/51 式断裂。"""
    raw = txt_path.read_text(encoding='utf-8-sig')
    parts = re.split(r'(?m)^(\d{1,3})\t', raw)
    entries: dict[int, str] = {}
    for i in range(1, len(parts), 2):
        cid, body = int(parts[i]), parts[i + 1].strip()
        if body.startswith('"') and body.endswith('"'):
            body = body[1:-1]
        body = re.sub(r'\s*\n\s*', '', body).strip()
        entries[cid] = body
    return entries


TN_PATTERNS = [
    ('latin_mixed', re.compile(r'[A-Za-z]{1,}\d+|\d+[A-Za-z]+'), '字母数字混合(工号TA1234/e招贷等),确认读法'),
    ('long_digits', re.compile(r'\d{5,}'), '长数字串(电话/卡号),通常逐位读'),
    ('decimal', re.compile(r'\d+\.\d+'), '小数(金额/利率),确认"点"读法'),
    ('percent', re.compile(r'\d+(?:\.\d+)?%|百分之[\dXxA-Za-z]+'), '百分比读法'),
    ('placeholder', re.compile(r'百分之X|XX|TODO|＿'), '占位符!录音实际念了什么需ASR确认'),
    ('wan_w', re.compile(r'\d+[Ww万]'), '万/W 单位读法'),
    ('date', re.compile(r'\d{1,2}月\d{1,2}[号日]|\d{4}年'), '日期读法'),
    ('paren', re.compile(r'[（(][^）)]*[）)]'), '括号内容: 确认是"真念出"还是"舞台提示"(后者要删)'),
    ('slash', re.compile(r'[\u4e00-\u9fff\w]{1,6}/[\u4e00-\u9fff\w]{1,6}'), '斜杠: 二选一(先生/女士)或读法(5万元/24个月), 需按实际发音核定'),
]


def tn_flags(text: str) -> list[dict]:
    flags = []
    for tag, pat, note in TN_PATTERNS:
        hits = pat.findall(text)
        if hits:
            flags.append({'type': tag, 'hits': hits[:8], 'note': note})
    return flags


_T2S = None


def _t2s(text: str) -> str:
    """繁体→简体(仅用于相似度比较, 不改原稿/manifest文本)。Whisper 常输出繁体造成假阳性。"""
    global _T2S
    try:
        if _T2S is None:
            from opencc import OpenCC
            _T2S = OpenCC('t2s')
        return _T2S.convert(text)
    except Exception:
        return text


def norm_for_cmp(text: str) -> str:
    """比较用归一化: 繁转简, 去标点/空白, 全角转半角, 小写。"""
    text = unicodedata.normalize('NFKC', _t2s(text))
    return ''.join(c.lower() for c in text
                   if unicodedata.category(c)[0] not in ('P', 'Z', 'C'))


def char_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_for_cmp(a), norm_for_cmp(b)).ratio()


# ---------------------------------------------------------------- 音频工具

def frame_db(x: np.ndarray, sr: int, frame_ms: int = 20) -> np.ndarray:
    n = int(sr * frame_ms / 1000)
    m = (len(x) // n) * n
    frames = x[:m].reshape(-1, n)
    rms = np.sqrt(np.maximum((frames ** 2).mean(axis=1), 1e-12))
    return 20 * np.log10(rms)


def edge_silence_ms(x: np.ndarray, sr: int):
    """返回 (首部静音ms, 尾部静音ms, 首个语音帧idx, 末个语音帧idx), 帧长20ms。"""
    db = frame_db(x, sr)
    idx = np.where(db > SILENCE_DB)[0]
    if len(idx) == 0:
        return len(x) / sr * 1000.0, 0.0, None, None
    return int(idx[0]) * 20.0, int(len(db) - 1 - idx[-1]) * 20.0, int(idx[0]), int(idx[-1])


def fix_edges(x: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """首尾静音规整到 EDGE_SILENCE_MS: 超过 MAX_EDGE_SILENCE_MS 裁到 300ms, 不足补零到 300ms。"""
    lead, trail, fi, li = edge_silence_ms(x, sr)
    info = {'lead_ms_in': round(lead, 1), 'trail_ms_in': round(trail, 1)}
    if fi is None:                      # 全静音, 原样返回(理论上已被排除)
        info.update({'lead_ms_out': None, 'trail_ms_out': None})
        return x, info
    n_edge = int(sr * EDGE_SILENCE_MS / 1000)
    n_frame = int(sr * 0.02)
    # 裁: 静音超 MAX 时, 从语音边界外扩 n_edge 下刀
    s = max(fi * n_frame - n_edge, 0) if lead > MAX_EDGE_SILENCE_MS else 0
    e = min((li + 1) * n_frame + n_edge, len(x)) if trail > MAX_EDGE_SILENCE_MS else len(x)
    y = x[s:e]
    # 补: 重测首尾静音, 不足 300ms 补零
    lead_now, trail_now, _, _ = edge_silence_ms(y, sr)
    pad_head = max(n_edge - int(lead_now / 1000 * sr), 0)
    pad_tail = max(n_edge - int(trail_now / 1000 * sr), 0)
    if pad_head or pad_tail:
        y = np.concatenate([np.zeros(pad_head, dtype=y.dtype), y,
                            np.zeros(pad_tail, dtype=y.dtype)])
    info.update({'lead_ms_out': round(max(lead_now, EDGE_SILENCE_MS)
                                      if lead <= MAX_EDGE_SILENCE_MS else EDGE_SILENCE_MS, 1),
                 'trail_ms_out': round(max(trail_now, EDGE_SILENCE_MS)
                                       if trail <= MAX_EDGE_SILENCE_MS else EDGE_SILENCE_MS, 1)})
    return y, info


def loudness_normalize(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """pyloudnorm 归一到 TARGET_LUFS; 失败(太短/静音)则不动。返回 (音频, 增益dB)。"""
    import pyloudnorm as pyln
    try:
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(x.astype(np.float64))
        if not np.isfinite(loud) or loud < -70:
            return x, 0.0
        gain_db = TARGET_LUFS - loud
        # 峰值保护: 归一后不超过 PEAK_CEIL
        peak_db = 20 * np.log10(max(np.abs(x).max(), 1e-12))
        gain_db = min(gain_db, PEAK_CEIL_DBFS - peak_db)
        return x * (10 ** (gain_db / 20)), gain_db
    except Exception:
        return x, 0.0


def resample_24k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR_OUT:
        return x
    import soxr
    return soxr.resample(x, sr, SR_OUT, quality='HQ')


def vad_segments(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """silero-vad 找语音区间(内部降到16k推理), 返回48k采样点区间列表。"""
    from silero_vad import load_silero_vad, get_speech_timestamps
    import soxr
    model = vad_segments._model  # noqa
    x16 = soxr.resample(x, sr, 16000, quality='HQ').astype(np.float32)
    ts = get_speech_timestamps(x16, model, sampling_rate=16000,
                               min_speech_duration_ms=250,
                               min_silence_duration_ms=250,
                               speech_pad_ms=100)
    scale = sr / 16000
    return [(int(t['start'] * scale), int(t['end'] * scale)) for t in ts]


vad_segments._model = None


def pack_segments(sp: list[tuple[int, int]], sr: int) -> list[tuple[int, int]]:
    """把语音区间打包成 ≤SEG_TARGET_S 的段, 只在静音间隙下刀, 不切语音内部。
    打包后处理碎段: <SEG_MIN_S 的段并入邻段(并后≤SEG_HARD_MAX_S), 并不进去则丢弃。"""
    if not sp:
        return []
    chunks, cur_s, cur_e = [], sp[0][0], sp[0][1]
    for s, e in sp[1:]:
        if (e - cur_s) / sr <= SEG_TARGET_S:
            cur_e = e
        else:
            chunks.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    chunks.append((cur_s, cur_e))

    merged: list[tuple[int, int]] = []
    for ch in chunks:
        dur = (ch[1] - ch[0]) / sr
        if dur >= SEG_MIN_S:
            merged.append(ch)
            continue
        if merged and (ch[1] - merged[-1][0]) / sr <= SEG_HARD_MAX_S:
            merged[-1] = (merged[-1][0], ch[1])       # 并入前段
        elif dur >= 1.0:
            merged.append(ch)                          # 并不了但≥1s, 保留待ASR定文本
        # <1s 又并不进去 → 丢弃(呼吸声/咔哒声)
    return merged


# ---------------------------------------------------------------- 阶段: text

def stage_text() -> dict[int, str]:
    texts = parse_corpus_texts(TEXT_TXT)
    OUT_DIR.mkdir(exist_ok=True)
    out = {str(k): {'text': v, 'tn_flags': tn_flags(v)} for k, v in texts.items()}
    (OUT_DIR / 'corpus_texts.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    n_flag = sum(1 for v in out.values() if v['tn_flags'])
    print(f'[text] 解析 {len(texts)} 条语料 → corpus_texts.json (含TN标记 {n_flag} 条)')
    return texts


# ---------------------------------------------------------------- 阶段: audio

def stage_audio():
    from silero_vad import load_silero_vad
    vad_segments._model = load_silero_vad()

    quality = {r['filename']: r for r in
               json.loads(QUALITY_JSON.read_text(encoding='utf-8'))['per_file']}
    wavs = sorted(DATA_DIR.glob('*.wav'))
    (OUT_DIR / 'wavs').mkdir(parents=True, exist_ok=True)
    for old in (OUT_DIR / 'wavs').glob('*.wav'):   # 重跑幂等: 清掉旧成品
        old.unlink()
    (OUT_DIR / 'excluded').mkdir(exist_ok=True)

    log, excluded_rows = [], []
    n_cut = 0
    for wp in wavs:
        parsed = parse_fname(wp.name)
        if parsed is None:
            continue
        take, cid = parsed
        if (take, cid) in EXCLUDE:
            shutil.copy2(wp, OUT_DIR / 'excluded' / wp.name)
            excluded_rows.append({'file': wp.name, 'reason': EXCLUDE[(take, cid)]})
            continue

        x, sr = sf.read(wp, dtype='float32')
        if x.ndim > 1:
            x = x.mean(axis=1)
        q = quality.get(wp.name, {})
        snr = q.get('snr_db')
        base_flags = {}
        if snr is not None and snr < SNR_BORDERLINE_DB:
            base_flags['snr_borderline'] = round(snr, 1)

        dur = len(x) / sr
        if dur > LONG_FILE_S:
            chunks = pack_segments(vad_segments(x, sr), sr)
            n_cut += 1
        else:
            chunks = [None]   # 不切

        for si, chunk in enumerate(chunks):
            seg = x if chunk is None else x[chunk[0]:chunk[1]]
            seg, edge_info = fix_edges(seg, sr)
            seg, gain_db = loudness_normalize(seg, sr)
            seg24 = resample_24k(seg, sr)
            peak_db = 20 * np.log10(max(np.abs(seg24).max(), 1e-12))
            name = (f'{take}-{cid}.wav' if chunk is None
                    else f'{take}-{cid}_s{si + 1:02d}.wav')
            sf.write(OUT_DIR / 'wavs' / name,
                     np.clip(seg24 * 32767, -32768, 32767).astype(np.int16),
                     SR_OUT, subtype='PCM_16')
            rec = {'out': name, 'src': wp.name, 'take': take, 'corpus_id': cid,
                   'seg_idx': None if chunk is None else si + 1,
                   'n_segs': len(chunks) if chunk is not None else 1,
                   'dur_in_s': round(dur, 2),
                   'dur_out_s': round(len(seg24) / SR_OUT, 2),
                   'gain_db': round(gain_db, 2),
                   'peak_dbfs_out': round(peak_db, 2),
                   'snr_db_in': snr, 'flags': base_flags}
            rec.update({k: v for k, v in edge_info.items()})
            log.append(rec)

    with open(OUT_DIR / 'audio_log.jsonl', 'w', encoding='utf-8') as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + '\n')
    (OUT_DIR / 'excluded' / '排除原因.json').write_text(
        json.dumps(excluded_rows, ensure_ascii=False, indent=1), encoding='utf-8')
    n_border = sum(1 for r in log if r['flags'].get('snr_borderline') is not None)
    print(f'[audio] 输出 {len(log)} 个成品 → wavs/ (24k/mono/-23LUFS)')
    print(f'        超30s切分 {n_cut} 个源文件; SNR边界标记 {n_border} 个; 排除 {len(excluded_rows)} 个')
    bad = [r['out'] for r in log if r['peak_dbfs_out'] > -0.5]
    if bad:
        print(f'        ⚠ 峰值过高: {bad}')


# ---------------------------------------------------------------- 阶段: asr

def split_sentences(text: str) -> list[str]:
    """按子句切分(全角句读 + ASCII逗号), 用于 VAD 段↔文本对齐。
    银行话术多为逗号长句, 只按 。！？； 切会导致子句数<段数(对齐退化)。
    小数点(.)不切, 保护 108071.79 等金额; 各子句拼接可无损还原原文。"""
    parts = re.split(r'(?<=[。！？；，、：,])', text)
    return [p for p in parts if p.strip()]


def align_segments_to_text(seg_asrs: list[str], text: str) -> list[str]:
    """DP: 把原稿子句按顺序分配给各切分段, 代价=1-相似度(段ASR文本 vs 分配子句串)。
    保证每段≥1子句、不产生空串、拼接可还原原文。"""
    sents = split_sentences(text)
    n_seg, n_sent = len(seg_asrs), len(sents)
    if n_seg == 1:
        return [text]
    if n_sent < n_seg:            # 子句仍不够分(极罕见): 按字符比例切, 不产生空串
        L = len(text)
        base = max(L // n_seg, 1)
        return [text[i * base:(i + 1) * base] if i < n_seg - 1 else text[i * base:]
                for i in range(n_seg)]
    INF = float('inf')
    # cost[i][j]: 前 i 段吃完前 j 句的最小代价; 每段至少1句
    cost = [[INF] * (n_sent + 1) for _ in range(n_seg + 1)]
    back = [[-1] * (n_sent + 1) for _ in range(n_seg + 1)]
    cost[0][0] = 0.0
    for i in range(1, n_seg + 1):
        for j in range(i, n_sent + 1):
            for k in range(i - 1, j):          # 第 i 段吃句子 k..j-1
                if cost[i - 1][k] == INF:
                    continue
                c = 1 - char_sim(seg_asrs[i - 1], ''.join(sents[k:j]))
                tot = cost[i - 1][k] + c
                if tot < cost[i][j]:
                    cost[i][j], back[i][j] = tot, k
    # 回溯
    assigns, j = [], n_sent
    for i in range(n_seg, 0, -1):
        k = back[i][j]
        assigns.append(''.join(sents[k:j]))
        j = k
    return assigns[::-1]


PAREN_RE = re.compile(r'[（(]([^）)]*)[）)]')


def strip_unspoken_parens(text: str, asr_concat: str) -> tuple[str, list[str]]:
    """ASR 佐证清理括号内容:
    - 括号内文字在 ASR 里找不到(没念) → 判为舞台提示, 整个括号删除(删多余文本对训练安全)
    - 括号内文字念了 → 保留内容, 去掉括号符号(括号是视觉标点不发音)
    返回 (清理后文本, 删除的舞台提示列表)。数字/占位符等不在此处理(交人工 TN)。"""
    asr_norm = norm_for_cmp(asr_concat)
    removed = []

    def repl(m):
        inner = m.group(1)
        key = norm_for_cmp(inner)[:5]           # 取括号内前5个有效字做存在性判断
        if key and key in asr_norm:
            return inner                        # 念了 → 去括号留内容
        removed.append(inner)
        return ''                               # 没念 → 删
    cleaned = PAREN_RE.sub(repl, text)
    return cleaned, removed



def build_asr_report(log: list, texts: dict, asr_of: dict):
    """按源文件分组做 VAD段↔文本对齐 + 相似度, 写 asr_report.jsonl。stage_asr/realign 共用。"""
    by_src = defaultdict(list)
    for rec in log:
        by_src[rec['src']].append(rec)

    rows = []
    for src, recs in by_src.items():
        cid = recs[0]['corpus_id']
        expected = texts.get(cid)
        if expected is None:
            for r in recs:
                rows.append({'out': r['out'], 'corpus_id': cid, 'expected': None,
                             'asr': asr_of.get(r['out'], ''), 'sim': None,
                             'status': 'no_text'})
            continue
        if len(recs) == 1:
            seg_texts = [expected]
        else:
            recs = sorted(recs, key=lambda r: r['seg_idx'])
            seg_texts = align_segments_to_text([asr_of.get(r['out'], '') for r in recs], expected)
        for r, st in zip(recs, seg_texts):
            sim = char_sim(st, asr_of.get(r['out'], ''))
            rows.append({'out': r['out'], 'corpus_id': cid, 'take': r['take'],
                         'seg_idx': r['seg_idx'],
                         'expected': st, 'asr': asr_of.get(r['out'], ''),
                         'sim': round(sim, 3),
                         'status': 'ok' if sim >= 0.75 else 'review'})

    rows.sort(key=lambda r: r['out'])
    with open(OUT_DIR / 'asr_report.jsonl', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + '\n')
    n_review = sum(1 for r in rows if r['status'] != 'ok')
    print(f'[asr] 报告 {len(rows)} 条, 相似度<0.75 待人工复核 {n_review} 条 → asr_report.jsonl')
    return rows


def stage_asr(model_size: str = 'small'):
    from faster_whisper import WhisperModel
    log = [json.loads(l) for l in
           (OUT_DIR / 'audio_log.jsonl').read_text(encoding='utf-8').splitlines()]
    texts = {int(k): v['text'] for k, v in
             json.loads((OUT_DIR / 'corpus_texts.json').read_text(encoding='utf-8')).items()}

    print(f'[asr] 加载 faster-whisper {model_size} (CPU int8, 首次运行自动下载)...')
    model = WhisperModel(model_size, device='cpu', compute_type='int8')

    def transcribe(path: Path) -> str:
        segs, _ = model.transcribe(str(path), language='zh', beam_size=5,
                                   condition_on_previous_text=False)
        return ''.join(s.text for s in segs).strip()

    # 先逐文件转写
    asr_of: dict[str, str] = {}
    for i, rec in enumerate(log):
        p = OUT_DIR / 'wavs' / rec['out']
        t = transcribe(p)
        asr_of[rec['out']] = t
        print(f'  [{i + 1}/{len(log)}] {rec["out"]}: {len(t)}字', flush=True)

    build_asr_report(log, texts, asr_of)


def stage_realign():
    """复用已缓存的 ASR 转写(asr_report.jsonl 的 asr 字段), 只重跑对齐+相似度。
    修对齐 bug 后无需重跑 30min 的 Whisper 转写。"""
    log = [json.loads(l) for l in
           (OUT_DIR / 'audio_log.jsonl').read_text(encoding='utf-8').splitlines()]
    texts = {int(k): v['text'] for k, v in
             json.loads((OUT_DIR / 'corpus_texts.json').read_text(encoding='utf-8')).items()}
    old = [json.loads(l) for l in
           (OUT_DIR / 'asr_report.jsonl').read_text(encoding='utf-8').splitlines()]
    asr_of = {r['out']: r.get('asr', '') for r in old}
    print(f'[realign] 复用 {len(asr_of)} 条缓存转写, 重跑对齐...')
    build_asr_report(log, texts, asr_of)



# ---------------------------------------------------------------- 阶段: finalize

def group_similar_texts(texts: dict[int, str], thr: float = 0.60) -> dict[int, int]:
    """近似重复文本归组(防 train/val 泄漏: 如 11⊃12+13, 37≡52, 41≡56)。返回 cid→组号。"""
    ids = sorted(texts)
    norm = {i: norm_for_cmp(texts[i]) for i in ids}
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for x, y in [(ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))]:
        nx, ny = norm[x], norm[y]
        if not nx or not ny:
            continue
        # 包含关系(一方是另一方子串)或高相似 → 同组
        if (nx in ny or ny in nx) and min(len(nx), len(ny)) > 15:
            union(x, y)
        elif abs(len(nx) - len(ny)) / max(len(nx), len(ny)) < 0.5:
            if SequenceMatcher(None, nx, ny).ratio() >= thr:
                union(x, y)
    return {i: find(i) for i in ids}


def stage_finalize(dev_ratio: float = 0.15, seed: int = 42):
    log = [json.loads(l) for l in
           (OUT_DIR / 'audio_log.jsonl').read_text(encoding='utf-8').splitlines()]
    corpus = json.loads((OUT_DIR / 'corpus_texts.json').read_text(encoding='utf-8'))
    texts = {int(k): v['text'] for k, v in corpus.items()}

    # asr 结果(可能不存在, 未跑 asr 阶段时 expected=原稿整段)
    asr_rep = {}
    asr_path = OUT_DIR / 'asr_report.jsonl'
    if asr_path.exists():
        for l in asr_path.read_text(encoding='utf-8').splitlines():
            r = json.loads(l)
            asr_rep[r['out']] = r

    # ---- 文本分组 → train/dev 拆分(同组同侧, 两遍录音天然同组=同文本) ----
    used_ids = sorted({r['corpus_id'] for r in log})
    gid = group_similar_texts({i: texts[i] for i in used_ids if i in texts})
    groups = defaultdict(list)
    for cid in used_ids:
        groups[gid.get(cid, cid)].append(cid)
    gkeys = sorted(groups)
    rng = np.random.default_rng(seed)
    n_files = len(log)
    # dev 优先纳入"小组"(多为独一无二短句)→ dev 多样、比例可控; 大模板组留 train(数据越多越好)。
    # 组大小相同则按语料号排序保证确定性; 同组所有语料号必须同侧(防近似重复文本泄漏)。
    def gsize(g):
        return sum(1 for r in log if r['corpus_id'] in groups[g])
    order = sorted(gkeys, key=lambda g: (gsize(g), g))
    dev_ids, cnt = set(), 0
    target = n_files * dev_ratio
    for g in order:
        gs = gsize(g)
        # 加入该组后不超过目标的 1.4 倍才纳入(避免单组把 dev 撑爆), 已达目标则停
        if cnt >= target:
            break
        if cnt + gs > target * 1.4 and cnt > 0:
            continue
        dev_ids.update(groups[g])
        cnt += gs
    print(f'[finalize] 文本组 {len(gkeys)} 个; dev 语料号 {sorted(dev_ids)} '
          f'({cnt}/{n_files} 文件, {cnt / n_files * 100:.0f}%)')

    # ---- manifest 草稿(字段对齐 prepare_aishell3.py; audio_codes/ref_mel 容器内填) ----
    # 预聚合每个语料号的全部 ASR 转写(两遍+各段), 用于佐证括号内容是否真念出
    corpus_asr = defaultdict(str)
    for r in log:
        corpus_asr[r['corpus_id']] += asr_rep.get(r['out'], {}).get('asr', '') + ' '

    removed_parens = {}   # cid → 删除的舞台提示
    rows_train, rows_dev = [], []
    for rec in log:
        cid = rec['corpus_id']
        ar = asr_rep.get(rec['out'], {})
        text = ar.get('expected') or texts.get(cid, '')
        text, removed = strip_unspoken_parens(text, corpus_asr.get(cid, ''))
        if removed:
            removed_parens.setdefault(cid, []).extend(removed)
        row = {
            'audio': f'wavs/{rec["out"]}',
            'text': text,
            'language': 'Chinese',
            'ref_audio': None,          # 容器内 prepare 时选定并预提 ref_mel
            'corpus_id': cid, 'take': rec['take'], 'seg_idx': rec['seg_idx'],
            'dur_s': rec['dur_out_s'], 'snr_db_in': rec['snr_db_in'],
            'flags': rec['flags'],
            'asr_sim': ar.get('sim'), 'asr_status': ar.get('status', 'not_run'),
            'audio_codes': None,        # TODO: 容器内 Qwen3TTSTokenizer 预编码后回填
        }
        (rows_dev if cid in dev_ids else rows_train).append(row)
    # 剔除无文本行(理论上不应发生)
    for rows, name in [(rows_train, 'train'), (rows_dev, 'dev')]:
        with open(OUT_DIR / f'{name}.jsonl', 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + '\n')
        print(f'  {name}.jsonl: {len(rows)} 行')

    # ---- TN 复核清单(带 ASR 证据 + 自动清理结果 + 人工核定要点) ----
    def review_note(fl_types, cid):
        notes = []
        if 'placeholder' in fl_types:
            notes.append('占位符(如百分之X): 看ASR实际念了什么数字, 替换X')
        if 'slash' in fl_types:
            notes.append('斜杠: 二选一(先生/女士取实际称呼)或读法(5万/24→五万除/分二十四), 按ASR核定')
        if 'paren' in fl_types:
            notes.append('括号: 已按ASR自动删未念的舞台提示/去念出内容的括号符号, 请复核')
        if 'long_digits' in fl_types:
            notes.append('长数字串(电话/卡号): 确认逐位读法, ASR数字不可靠需人耳')
        if 'decimal' in fl_types or 'percent' in fl_types:
            notes.append('小数/百分比(金额/利率): 确认"点/百分之"读法, 以原稿数字为准(ASR易错)')
        if 'wan_w' in fl_types:
            notes.append('万/W单位: 确认读"万"')
        if 'date' in fl_types:
            notes.append('日期: 确认"X月X号"读法')
        return ' | '.join(notes)

    with open(OUT_DIR / 'tn_review.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['corpus_id', 'TN问题类型', '命中内容', '原稿文本',
                    'ASR实际转写(证据,数字不可靠)', '自动清理后文本(删舞台提示)',
                    '人工核定要点', '最终核定文本(人工填写)'])
        for cid in sorted({r['corpus_id'] for r in log}):
            fl = corpus.get(str(cid), {}).get('tn_flags', [])
            if not fl:
                continue
            asr_txts = [asr_rep[r['out']]['asr'] for r in log
                        if r['corpus_id'] == cid and r['out'] in asr_rep]
            cleaned, _ = strip_unspoken_parens(texts.get(cid, ''), corpus_asr.get(cid, ''))
            w.writerow([cid,
                        '; '.join(x['type'] for x in fl),
                        ' | '.join(','.join(map(str, x['hits'])) for x in fl),
                        texts.get(cid, ''),
                        ' ⏎ '.join(asr_txts),
                        cleaned,
                        review_note({x['type'] for x in fl}, cid),
                        ''])
    print(f'  tn_review.csv: TN 人工复核清单')
    if removed_parens:
        print(f'  自动删除舞台提示(ASR确认未念): '
              + '; '.join(f'语料{c}「{"、".join(set(v))}」' for c, v in removed_parens.items()))

    # ---- 听辨清单 HTML(SNR 边界值) ----
    borderline = [r for r in log if r['flags'].get('snr_borderline') is not None]
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>SNR边界值听辨清单</title>',
            '<style>body{font-family:sans-serif;max-width:900px;margin:20px auto}',
            'tr.excluded{opacity:.4;text-decoration:line-through}',
            'td,th{border:1px solid #ccc;padding:6px}button{margin:10px 0;padding:8px}</style></head><body>',
            f'<h2>SNR 边界值听辨({len(borderline)} 个成品文件)</h2>',
            '<p>逐条播放: 听感可接受→不动; 有明显嘶嘶声/环境噪声→勾选"排除"。完成后点导出按钮, 把下载的 JSON 交给助手脚本过滤 manifest。</p>',
            '<button onclick="exp()">导出排除清单 JSON</button>',
            '<table><tr><th>排除</th><th>文件</th><th>SNR(dB)</th><th>文本</th></tr>']
    for r in sorted(borderline, key=lambda x: x['snr_db_in']):
        cid = r['corpus_id']
        t = (texts.get(cid, '') or '')[:60]
        html.append(
            f'<tr id="row_{r["out"]}"><td><input type="checkbox" f="{r["out"]}" '
            f'onchange="this.closest(\'tr\').classList.toggle(\'excluded\',this.checked)"></td>'
            f'<td><audio controls preload="none" src="wavs/{r["out"]}" style="height:32px"></audio>'
            f'<br>{r["out"]}</td><td>{r["snr_db_in"]}</td><td>{t}</td></tr>')
    html.append('''</table><script>
function exp(){const fs=[...document.querySelectorAll('input:checked')].map(c=>c.getAttribute('f'));
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({excluded:fs},null,1)],{type:'application/json'}));
a.download='听辨排除清单.json';a.click();}</script></body></html>''')
    (OUT_DIR / '听辨清单.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'  听辨清单.html: {len(borderline)} 条待听辨')

    # ---- 处理报告 ----
    total_out = sum(r['dur_out_s'] for r in log)
    # ASR 相似度统计
    sims = sorted(r['sim'] for r in asr_rep.values() if r.get('sim') is not None)
    if sims:
        import statistics
        median_sim = statistics.median(sims)
        n_sim_ok = sum(s >= 0.75 for s in sims)
        n_sim_tot = len(sims)
    else:
        median_sim, n_sim_ok, n_sim_tot = 0.0, 0, 0
    # 关键 TN 发现(占位符/二选一/斜杠读法/舞台提示): 原稿写法≠实际念法, 必须人工核定
    tn_findings = []
    KEY_TYPES = {'placeholder', 'slash', 'paren'}
    for cid in sorted({r['corpus_id'] for r in log}):
        fl = corpus.get(str(cid), {}).get('tn_flags', [])
        key_fl = [x for x in fl if x['type'] in KEY_TYPES]
        if not key_fl:
            continue
        asr_txts = [asr_rep[r['out']]['asr'] for r in log
                    if r['corpus_id'] == cid and r['out'] in asr_rep]
        hits = '; '.join(f"{x['type']}:{'/'.join(map(str, x['hits'][:3]))}" for x in key_fl)
        ev = (asr_txts[0][:60] + '…') if asr_txts else '(无ASR)'
        tn_findings.append(f'- 语料{cid} [{hits}] 原稿「{texts.get(cid, "")[:50]}…」 ASR证据「{ev}」')
    if removed_parens:
        tn_findings.append('- 已自动删除(ASR确认未念的舞台提示): '
                           + '; '.join(f'语料{c}「{"、".join(sorted(set(v)))}」'
                                       for c, v in removed_parens.items()))
    if not tn_findings:
        tn_findings = ['(无占位符/斜杠/括号类关键问题)']
    rep = [
        '# 客户数据集后处理报告', '',
        f'- 源文件: 162 WAV / 50.5 min → 排除 {len(EXCLUDE)} 条(docs/09 §4)',
        f'- 成品: **{len(log)} 个 / {total_out / 60:.1f} min**, 24kHz mono 16bit, {TARGET_LUFS:.0f} LUFS, 首尾静音 {EDGE_SILENCE_MS}ms',
        f'- VAD 切分: {sum(1 for r in log if (r["seg_idx"] or 0) > 0)} 段(来自超30s源文件)',
        f'- train/dev = {len(rows_train)}/{len(rows_dev)} (按文本组拆分, 近似重复文本同组同侧防泄漏; 同一原稿两遍录音文本相同天然同侧)',
        f'- SNR 边界待听辨: {len(borderline)} 条 → 听辨清单.html',
        f'- ASR 校验: {"已跑, 见 asr_report.jsonl" if asr_path.exists() else "未跑(--stage asr)"}',
        f'- 文本相似度: 中位 {median_sim:.2f}, {n_sim_ok}/{n_sim_tot} 条≥0.75 → 录音高度忠实原稿',
        '', '## 关键 TN 发现(必须人工核定后回填 manifest text)',
        '> 原稿书面写法 ≠ 录音实际念法。以下语料的 text 字段直接拿去训练会教模型读错, **必须按 ASR 证据人工核定**。',
        '> 完整 41 条见 tn_review.csv; 下面列最关键的"占位符/二选一/斜杠读法"类。数字类以原稿为准(Whisper 对精确数字不可靠)。',
        '',
    ] + tn_findings + [
        '', '## 待人工步骤',
        '1. 听辨清单.html 过一遍 → 导出排除 JSON → 用 `--apply_exclusion` 过滤 manifest(见脚本 main)',
        '2. tn_review.csv 核定数字/工号/占位符/斜杠读法 → 修订文本后回填 manifest 的 text 字段',
        '3. asr_report.jsonl 里 status=review 的条目人工听一下(可能是漏读/错读/ASR自身错误)',
        '', '## 顺延到 NPU 容器',
        '- audio_codes 预编码 + ref_mel 提取: 用 prepare 脚本(参照 prepare_aishell3.py)读本目录 train/dev.jsonl 回填',
    ]
    (OUT_DIR / '处理报告.md').write_text('\n'.join(rep), encoding='utf-8')
    print('  处理报告.md')


def apply_exclusion(json_path: Path):
    """听辨完成后: 按排除清单从 train/dev.jsonl 过滤, 并移除对应 wav 引用(文件保留在盘上)。"""
    exc = set(json.loads(json_path.read_text(encoding='utf-8'))['excluded'])
    for name in ('train', 'dev'):
        p = OUT_DIR / f'{name}.jsonl'
        rows = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
        keep = [r for r in rows if Path(r['audio']).name not in exc]
        with open(p, 'w', encoding='utf-8') as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + '\n')
        print(f'{name}.jsonl: {len(rows)} → {len(keep)} (排除 {len(rows) - len(keep)})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', default='all',
                    choices=['all', 'text', 'audio', 'asr', 'realign', 'finalize'])
    ap.add_argument('--whisper_model', default='small')
    ap.add_argument('--dev_ratio', type=float, default=0.15)
    ap.add_argument('--apply_exclusion', default=None,
                    help='听辨排除清单 JSON 路径, 直接从 train/dev.jsonl 过滤')
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    if args.apply_exclusion:
        apply_exclusion(Path(args.apply_exclusion))
        return
    if args.stage in ('all', 'text'):
        stage_text()
    if args.stage in ('all', 'audio'):
        stage_audio()
    if args.stage in ('all', 'asr'):
        stage_asr(args.whisper_model)
    if args.stage == 'realign':
        stage_realign()
    if args.stage in ('all', 'realign', 'finalize'):
        stage_finalize(dev_ratio=args.dev_ratio)


if __name__ == '__main__':
    main()
