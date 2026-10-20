#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS 微调数据集质量自动分析脚本
=================================
对照"旧标准"(客户收到的 8 条录制要求) + "新标准"(docs/03)关键项，
自动分析客户录制的数据集是否满足后续 TTS 模型微调要求。

分析项:
  1. 格式合规    (旧标准 #3: 48kHz/16bit/WAV)
  2. DC偏移      (旧标准 #1: 零电平无漂移 < ±0.5%FS)
  3. 削波与峰值  (旧标准 #1: 幅度无截幅, 峰值不到 0 dBFS)
  4. 首尾静音    (旧标准 #1: 首尾静音 ≥ 300ms)
  5. 信噪比 SNR  (旧标准 #2: SNR ≥ 35dB)
  6. 背景噪声    (旧标准 #2: 背景噪声 < -60dB)
  7. 单条时长    (旧标准 #4: ≤ 30秒, 最好 < 25秒)
  8. 总时长评估  (新标准: 推荐 15~30 分钟)
  9. 文件命名    (旧标准 #4: 每条语料读2遍→2个文件)
  10. 音量一致性 (旧标准 #7: 避免前后音量差异明显)
  11. 与新标准差异标注

输出:
  - 终端汇总
  - DATASET_DIR/数据集质量分析报告.txt   (人类可读)
  - DATASET_DIR/数据集质量分析报告.json   (机器可读)

用法:
  python scripts/analyze_dataset_quality.py
  python scripts/analyze_dataset_quality.py --data_dir "D:\\path\\to\\dataset"

依赖: numpy, soundfile  (可选: scipy, librosa —— 已安装则自动用)
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sys.exit("错误: 需要 soundfile 库。请运行: pip install soundfile")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════
# 标准阈值
# ═══════════════════════════════════════════════════════════════════════

# 旧标准（客户收到的 8 条录制要求）
OLD = {
    "sr_hz": 48000,          # #3 存储: 48kHz
    "bit_depth": 16,          # #3 存储: 16bit
    "max_dur_s": 30.0,        # #4 时长: ≤30秒
    "ideal_max_dur_s": 25.0,  # #4 时长: 最好<25秒
    "min_silence_ms": 300,    # #1 首尾静音≥300ms
    "min_snr_db": 35.0,       # #2 SNR≥35dB
    "max_noise_dbfs": -60.0,  # #2 背景噪声<-60dB
    "dc_max_pct": 0.5,        # #1 零电平无漂移 <±0.5%FS
    "clip_dbfs": -0.5,        # 峰值 >-0.5dBFS 判为削波风险
}

# 新标准（docs/03-数据集录制标准与方案.md，客户尚未收到）
NEW = {
    "master_sr": 48000,
    "master_bit": 24,
    "master_ch": 2,            # stereo 母带
    "qwen3_sr": 24000,         # Qwen3-TTS 硬要求
    "qwen3_ch": 1,             # mono
    "moss_sr": 48000,          # MOSS 原生
    "moss_ch": 2,              # stereo
    "min_clip_s": 5.0,         # 单条 5~15 秒
    "max_clip_s": 15.0,
    "total_min_min": 15,       # 总时长 15~30 分钟
    "total_max_min": 30,
    "peak_low": -12,           # 峰值 -6~-12 dBFS
    "peak_high": -6,
    "peak_max": -3,            # 最大不超 -3 dBFS
    "lufs": -23.0,             # 响度归一 -23 LUFS
    "dnsmos_min": 3.0,         # DNSMOS ≥3
    "rec_silence_ms": 500,     # 录制留 500ms~1s
    "prod_silence_ms": 300,   # 成品裁 300ms
}

# 分析参数
P = {
    "frame_ms": 20,            # 帧长 20ms
    "hop_ms": 10,              # 步长 10ms
    "silence_db": -40.0,       # 静音判定阈值 (dBFS)
    "noise_pct": 10,           # 低 10% 帧作为噪声估计
    "signal_pct": 90,          # 高 90% 帧作为信号估计
}


# ═══════════════════════════════════════════════════════════════════════
# 单文件分析
# ═══════════════════════════════════════════════════════════════════════

def analyze_wav(filepath: Path) -> dict:
    """分析单个 WAV 文件，返回指标字典"""
    r = {"filename": filepath.name, "filepath": str(filepath)}
    try:
        info = sf.info(str(filepath))
        audio, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        n_ch = audio.shape[1]
        n = audio.shape[0]
        dur = n / sr

        # 降混到 mono 用于分析
        mono = audio.mean(axis=1) if n_ch > 1 else audio[:, 0]

        # ── 格式信息 ──
        sub = info.subtype
        if "PCM_" in sub:
            bit = int(sub.split("_")[1])
        elif "FLOAT" in sub:
            bit = 32
        elif "DOUBLE" in sub:
            bit = 64
        else:
            bit = None
        r["sr"] = sr
        r["bit"] = bit
        r["channels"] = n_ch
        r["subtype"] = sub
        r["dur_s"] = round(dur, 3)

        # ── DC 偏移 ──
        dc = float(np.mean(mono))
        r["dc_offset"] = dc
        r["dc_pct"] = round(abs(dc) * 100, 4)

        # ── 峰值与削波 ──
        peak = float(np.max(np.abs(mono)))
        peak_db = 20 * np.log10(peak + 1e-12)
        clip_thresh = 10 ** (-0.5 / 20)   # -0.5 dBFS
        n_clip = int(np.sum(np.abs(mono) >= clip_thresh))
        r["peak_dbfs"] = round(peak_db, 2)
        r["n_clipped"] = n_clip
        r["clip_ratio"] = round(n_clip / n, 6)
        r["is_clipping"] = peak_db > OLD["clip_dbfs"]

        # ── RMS (整条) ──
        rms = float(np.sqrt(np.mean(mono ** 2)))
        r["rms_dbfs"] = round(20 * np.log10(rms + 1e-12), 2)

        # ── 帧分析 ──
        flen = max(1, int(sr * P["frame_ms"] / 1000))
        hlen = max(1, int(sr * P["hop_ms"] / 1000))
        nframes = max(1, (n - flen) // hlen + 1)
        fr_rms = np.empty(nframes)
        for i in range(nframes):
            s = i * hlen
            e = min(s + flen, n)
            fr_rms[i] = np.sqrt(np.mean(mono[s:e] ** 2))
        fr_db = 20 * np.log10(fr_rms + 1e-12)

        # ── 首尾静音 ──
        speech_idx = np.where(fr_db > P["silence_db"])[0]
        if len(speech_idx) > 0:
            lead_ms = speech_idx[0] * hlen / sr * 1000
            trail_ms = (n - (speech_idx[-1] * hlen + flen)) / sr * 1000
            r["speech_ratio"] = round(len(speech_idx) / nframes, 4)
        else:
            lead_ms = dur * 1000
            trail_ms = 0.0
            r["speech_ratio"] = 0.0
            r["warning"] = "no_speech_detected"
        r["lead_ms"] = round(max(0.0, lead_ms), 1)
        r["trail_ms"] = round(max(0.0, trail_ms), 1)

        # ── SNR ──
        sr_sorted = np.sort(fr_rms)
        noise_rms = float(np.percentile(sr_sorted, P["noise_pct"]))
        signal_rms = float(np.percentile(sr_sorted, P["signal_pct"]))
        noise_db = 20 * np.log10(noise_rms + 1e-12)
        if noise_rms > 1e-10:
            snr = 20 * np.log10(signal_rms / (noise_rms + 1e-10))
        else:
            snr = 999.0
        r["noise_dbfs"] = round(noise_db, 2)
        r["snr_db"] = round(snr, 2)

        r["status"] = "ok"
    except Exception as e:
        r["status"] = "error"
        r["error"] = str(e)
    return r


# ═══════════════════════════════════════════════════════════════════════
# 文件名解析
# ═══════════════════════════════════════════════════════════════════════

def parse_fname(fname: str):
    """解析 'X - Y.wav' 格式, 返回 (group, number)；不匹配返回 (None, None)"""
    m = re.match(r'^(\d+)\s*-\s*(\d+)\.wav$', fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# ═══════════════════════════════════════════════════════════════════════
# 数据集级分析
# ═══════════════════════════════════════════════════════════════════════

def analyze_dataset(data_dir: Path):
    wavs = sorted(data_dir.glob("*.wav"))
    print(f"\n找到 {len(wavs)} 个 WAV 文件，开始分析...\n")

    results = []
    for i, w in enumerate(wavs):
        r = analyze_wav(w)
        results.append(r)
        if (i + 1) % 20 == 0 or i + 1 == len(wavs):
            print(f"  进度: {i+1}/{len(wavs)} ({(i+1)*100//len(wavs)}%)")

    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    summary = compute_summary(ok, wavs)
    return results, summary, err


def compute_summary(ok, wavs):
    s = {}
    s["n_files"] = len(wavs)
    s["n_ok"] = len(ok)
    s["n_err"] = len(wavs) - len(ok)

    # ── 格式统计 ──
    srs = [r["sr"] for r in ok]
    bits = [r["bit"] for r in ok]
    chs = [r["channels"] for r in ok]
    s["sr_set"] = sorted(set(srs))
    s["bit_set"] = sorted(set(bits))
    s["ch_set"] = sorted(set(chs))
    s["sr_all_48k"] = all(x == OLD["sr_hz"] for x in srs) if srs else False
    s["bit_all_16"] = all(x == OLD["bit_depth"] for x in bits) if bits else False
    s["ch_all_mono"] = all(x == 1 for x in chs) if chs else False

    # ── 时长 ──
    durs = [r["dur_s"] for r in ok]
    if durs:
        s["total_dur_s"] = round(sum(durs), 1)
        s["total_dur_min"] = round(sum(durs) / 60, 1)
        s["dur_mean"] = round(float(np.mean(durs)), 2)
        s["dur_median"] = round(float(np.median(durs)), 2)
        s["dur_min"] = round(min(durs), 2)
        s["dur_max"] = round(max(durs), 2)
        s["dur_std"] = round(float(np.std(durs)), 2)
        s["n_over_30s"] = sum(1 for d in durs if d > OLD["max_dur_s"])
        s["n_over_25s"] = sum(1 for d in durs if d > OLD["ideal_max_dur_s"])
        s["over_30s_files"] = [r["filename"] for r in ok if r["dur_s"] > OLD["max_dur_s"]]
        s["over_25s_files"] = [r["filename"] for r in ok if r["dur_s"] > OLD["ideal_max_dur_s"]]
        s["under_3s_files"] = [r["filename"] for r in ok if r["dur_s"] < 3.0]
    else:
        s["total_dur_s"] = 0
        s["total_dur_min"] = 0
        s["dur_mean"] = 0
        s["dur_median"] = 0
        s["dur_min"] = 0
        s["dur_max"] = 0
        s["dur_std"] = 0
        s["n_over_30s"] = 0
        s["n_over_25s"] = 0
        s["over_30s_files"] = []
        s["over_25s_files"] = []
        s["under_3s_files"] = []

    # ── DC 偏移 ──
    dcs = [r["dc_pct"] for r in ok]
    s["dc_mean"] = round(float(np.mean(dcs)), 4) if dcs else 0
    s["dc_max"] = round(max(dcs), 4) if dcs else 0
    s["n_dc_fail"] = sum(1 for d in dcs if d > OLD["dc_max_pct"])
    s["dc_fail_files"] = [r["filename"] for r in ok if r["dc_pct"] > OLD["dc_max_pct"]]

    # ── 削波 ──
    s["n_clipping"] = sum(1 for r in ok if r["is_clipping"])
    s["clip_files"] = [r["filename"] for r in ok if r["is_clipping"]]
    peaks = [r["peak_dbfs"] for r in ok]
    s["peak_mean"] = round(float(np.mean(peaks)), 2) if peaks else 0
    s["peak_max"] = round(max(peaks), 2) if peaks else 0
    s["peak_min"] = round(min(peaks), 2) if peaks else 0

    # ── 首尾静音 ──
    leads = [r["lead_ms"] for r in ok]
    trails = [r["trail_ms"] for r in ok]
    s["lead_mean"] = round(float(np.mean(leads)), 1) if leads else 0
    s["lead_min"] = round(min(leads), 1) if leads else 0
    s["lead_max"] = round(max(leads), 1) if leads else 0
    s["trail_mean"] = round(float(np.mean(trails)), 1) if trails else 0
    s["trail_min"] = round(min(trails), 1) if trails else 0
    s["trail_max"] = round(max(trails), 1) if trails else 0
    s["n_lead_fail"] = sum(1 for x in leads if x < OLD["min_silence_ms"])
    s["n_trail_fail"] = sum(1 for x in trails if x < OLD["min_silence_ms"])
    s["lead_fail_files"] = [r["filename"] for r in ok if r["lead_ms"] < OLD["min_silence_ms"]]
    s["trail_fail_files"] = [r["filename"] for r in ok if r["trail_ms"] < OLD["min_silence_ms"]]

    # ── SNR ──
    snrs = [r["snr_db"] for r in ok]
    s["snr_mean"] = round(float(np.mean(snrs)), 1) if snrs else 0
    s["snr_median"] = round(float(np.median(snrs)), 1) if snrs else 0
    s["snr_min"] = round(min(snrs), 1) if snrs else 0
    s["snr_max"] = round(max(snrs), 1) if snrs else 0
    s["n_snr_fail"] = sum(1 for x in snrs if x < OLD["min_snr_db"])
    s["snr_fail_files"] = [r["filename"] for r in ok if r["snr_db"] < OLD["min_snr_db"]]

    # ── 噪声 ──
    noises = [r["noise_dbfs"] for r in ok]
    s["noise_mean"] = round(float(np.mean(noises)), 1) if noises else 0
    s["noise_max"] = round(max(noises), 1) if noises else 0
    s["noise_min"] = round(min(noises), 1) if noises else 0
    s["n_noise_fail"] = sum(1 for x in noises if x > OLD["max_noise_dbfs"])
    s["noise_fail_files"] = [r["filename"] for r in ok if r["noise_dbfs"] > OLD["max_noise_dbfs"]]

    # ── 静音文件检测 (全零或近静音) ──
    SILENCE_DBFS = -60.0
    silent_files = [r for r in ok if r["peak_dbfs"] < SILENCE_DBFS]
    s["n_silent"] = len(silent_files)
    s["silent_files"] = [r["filename"] for r in silent_files]

    # ── RMS 一致性 (排除静音文件) ──
    valid = [r for r in ok if r["peak_dbfs"] >= SILENCE_DBFS]
    rmss = [r["rms_dbfs"] for r in valid]
    s["n_valid"] = len(valid)
    s["rms_mean"] = round(float(np.mean(rmss)), 2) if rmss else 0
    s["rms_std"] = round(float(np.std(rmss)), 2) if rmss else 0
    s["rms_min"] = round(min(rmss), 2) if rmss else 0
    s["rms_max"] = round(max(rmss), 2) if rmss else 0
    s["rms_range"] = round(max(rmss) - min(rmss), 2) if rmss else 0

    # ── 命名 & 配对 ──
    groups = defaultdict(set)
    for r in ok:
        g, num = parse_fname(r["filename"])
        if g is not None:
            groups[str(g)].add(num)
    s["groups"] = {g: sorted(nums) for g, nums in groups.items()}
    s["naming_pattern"] = "'X - Y.wav'" if groups else "unrecognized"

    # 配对检查
    if len(groups) >= 2:
        all_nums = set()
        for nums in groups.values():
            all_nums.update(nums)
        s["all_corpus_nums"] = sorted(all_nums)
        for g in sorted(groups.keys()):
            nums = groups[g]
            s[f"group_{g}_count"] = len(nums)
            s[f"group_{g}_range"] = f"{min(nums)}~{max(nums)}" if nums else "empty"
        # 缺失编号
        if all_nums:
            full = set(range(min(all_nums), max(all_nums) + 1))
            s["missing_nums"] = sorted(full - all_nums)
        # 单边编号
        group_keys = sorted(groups.keys())
        for g in group_keys:
            others = set()
            for g2 in group_keys:
                if g2 != g:
                    others.update(groups[g2])
            only_g = sorted(groups[g] - others)
            s[f"group_{g}_only"] = only_g
        # 配对完整
        has_missing = bool(s.get("missing_nums"))
        has_single = any(s.get(f"group_{g}_only") for g in group_keys)
        s["pairing_complete"] = not has_missing and not has_single
    else:
        s["all_corpus_nums"] = []
        s["missing_nums"] = []
        s["pairing_complete"] = None

    return s


# ═══════════════════════════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════════════════════════

def fmt_list(lst, max_show=15):
    """格式化列表为字符串，超长截断"""
    if not lst:
        return "（无）"
    items = [str(x) for x in lst]
    if len(items) <= max_show:
        return ", ".join(items)
    return ", ".join(items[:max_show]) + f" ... (共 {len(items)} 项)"


def generate_report(results, s, data_dir):
    L = []
    w = lambda text="": L.append(text)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w("=" * 80)
    w("  TTS 微调数据集质量自动分析报告")
    w(f"  生成时间: {now}")
    w(f"  数据集路径: {data_dir}")
    w("=" * 80)
    w()

    # ── 一、概要 ──
    w("一、概要")
    w("-" * 70)
    w(f"  文件总数:         {s['n_files']}")
    w(f"  成功分析:         {s['n_ok']}")
    w(f"  分析失败:         {s['n_err']}")
    w(f"  总时长:           {s['total_dur_s']:.1f} 秒 = {s['total_dur_min']:.1f} 分钟")
    w(f"  单条平均时长:     {s['dur_mean']:.1f} 秒")
    w(f"  单条最短:         {s['dur_min']:.1f} 秒")
    w(f"  单条最长:         {s['dur_max']:.1f} 秒")
    w()

    # ── 二、格式合规 ──
    w("二、格式合规（旧标准 #3: 48kHz / 16bit / WAV）")
    w("-" * 70)
    sr_ok = s["sr_all_48k"]
    bit_ok = s["bit_all_16"]
    ch_mono = s["ch_all_mono"]
    w(f"  采样率:           {s['sr_set']} → {'✅ 合格 (48kHz)' if sr_ok else '❌ 不合格'}")
    w(f"  位深:             {s['bit_set']} → {'✅ 合格 (16bit)' if bit_ok else '❌ 不合格'}")
    w(f"  声道:             {s['ch_set']} → {'单声道 (mono)' if ch_mono else '多声道'}")
    w(f"  文件格式:         WAV ✅")
    w()
    w("  【新标准对比 (docs/03)】")
    w(f"    新标准推荐母带: 48kHz / 24-bit / stereo (立体声)")
    w(f"    当前数据:       48kHz / {s['bit_set']} / {'mono' if ch_mono else s['ch_set']}")
    if ch_mono:
        w(f"    ⚠ 单声道: Qwen3-TTS 需 mono ✅, 但 MOSS-TTS 原生 stereo ❌")
        w(f"      → 若只主攻 Qwen3-TTS, mono 完全够用")
        w(f"      → 若将来要用 MOSS, 需重新录 stereo 母带")
    if bit_ok:
        w(f"    ⚠ 16-bit: 新标准推荐 24-bit 母带留余量; 16bit 可用但剪辑/降噪空间小")
    w()

    # ── 三、DC偏移 ──
    w("三、DC偏移 / 零电平漂移（旧标准 #1: 零电平无漂移）")
    w("-" * 70)
    w(f"  平均 DC 偏移:     {s['dc_mean']:.4f}%FS  (阈值 < ±0.5%FS)")
    w(f"  最大 DC 偏移:     {s['dc_max']:.4f}%FS")
    w(f"  不达标条数:       {s['n_dc_fail']}")
    if s["n_dc_fail"] > 0:
        w(f"  不达标文件:       {fmt_list(s['dc_fail_files'])}")
    status = "✅ 全部合格" if s["n_dc_fail"] == 0 else f"❌ {s['n_dc_fail']} 条不达标"
    w(f"  结论:             {status}")
    w()

    # ── 四、削波与峰值 ──
    w("四、削波与峰值电平（旧标准 #1: 幅度无截幅）")
    w("-" * 70)
    w(f"  峰值均值:         {s['peak_mean']:.2f} dBFS")
    w(f"  峰值最大:         {s['peak_max']:.2f} dBFS")
    w(f"  峰值最小:         {s['peak_min']:.2f} dBFS")
    w(f"  削波条数:         {s['n_clipping']}")
    if s["n_clipping"] > 0:
        w(f"  削波文件:         {fmt_list(s['clip_files'])}")
    status = "✅ 无削波" if s["n_clipping"] == 0 else f"❌ {s['n_clipping']} 条削波"
    w(f"  结论:             {status}")
    w(f"  【新标准对比】峰值理想区间 -6~-12 dBFS, 最大不超 -3 dBFS")
    if s["peak_max"] > NEW["peak_max"]:
        w(f"    ❌ 最大峰值 {s['peak_max']:.2f} dBFS 超 -3, 有削波风险")
    elif s["peak_max"] > NEW["peak_high"]:
        w(f"    ⚠ 最大峰值 {s['peak_max']:.2f} dBFS 超 -6, 偏高")
    w()

    # ── 五、首尾静音 ──
    w("五、首尾静音（旧标准 #1: 首尾静音 ≥ 300ms）")
    w("-" * 70)
    w(f"  首部静音均值:     {s['lead_mean']:.0f} ms")
    w(f"  首部静音最短:     {s['lead_min']:.0f} ms  (阈值 ≥ 300ms)")
    w(f"  首部静音最长:     {s['lead_max']:.0f} ms")
    w(f"  首部不达标:       {s['n_lead_fail']} 条")
    if s["n_lead_fail"] > 0:
        w(f"  首部不达标文件:   {fmt_list(s['lead_fail_files'])}")
    w(f"  尾部静音均值:     {s['trail_mean']:.0f} ms")
    w(f"  尾部静音最短:     {s['trail_min']:.0f} ms  (阈值 ≥ 300ms)")
    w(f"  尾部静音最长:     {s['trail_max']:.0f} ms")
    w(f"  尾部不达标:       {s['n_trail_fail']} 条")
    if s["n_trail_fail"] > 0:
        w(f"  尾部不达标文件:   {fmt_list(s['trail_fail_files'])}")
    n_sil = s["n_lead_fail"] + s["n_trail_fail"]
    status = "✅ 全部达标" if n_sil == 0 else f"❌ {n_sil} 条不达标"
    w(f"  结论:             {status}")
    w(f"  【新标准对比】录制留 500ms~1s 静音, 成品统一裁 300ms")
    w()

    # ── 六、SNR ──
    w("六、信噪比 SNR（旧标准 #2: SNR ≥ 35dB）")
    w("-" * 70)
    w(f"  SNR 均值:         {s['snr_mean']:.1f} dB")
    w(f"  SNR 中位数:       {s['snr_median']:.1f} dB")
    w(f"  SNR 最低:         {s['snr_min']:.1f} dB  (阈值 ≥ 35dB)")
    w(f"  SNR 最高:         {s['snr_max']:.1f} dB")
    w(f"  不达标条数:       {s['n_snr_fail']}")
    if s["n_snr_fail"] > 0:
        w(f"  不达标文件:       {fmt_list(s['snr_fail_files'])}")
    status = "✅ 全部达标" if s["n_snr_fail"] == 0 else f"❌ {s['n_snr_fail']} 条不达标"
    w(f"  结论:             {status}")
    w()

    # ── 七、背景噪声 ──
    w("七、背景噪声（旧标准 #2: 背景噪声 < -60dB）")
    w("-" * 70)
    w(f"  噪声均值:         {s['noise_mean']:.1f} dBFS")
    w(f"  噪声最大:         {s['noise_max']:.1f} dBFS  (阈值 < -60dBFS)")
    w(f"  噪声最小:         {s['noise_min']:.1f} dBFS")
    w(f"  不达标条数:       {s['n_noise_fail']}")
    if s["n_noise_fail"] > 0:
        w(f"  不达标文件:       {fmt_list(s['noise_fail_files'])}")
    status = "✅ 全部达标" if s["n_noise_fail"] == 0 else f"❌ {s['n_noise_fail']} 条不达标"
    w(f"  结论:             {status}")
    w()

    # ── 八、单条时长 ──
    w("八、单条时长（旧标准 #4: ≤ 30秒, 最好 < 25秒）")
    w("-" * 70)
    w(f"  超 30 秒条数:     {s['n_over_30s']}")
    if s["n_over_30s"] > 0:
        w(f"  超 30 秒文件:     {fmt_list(s['over_30s_files'])}")
    w(f"  超 25 秒条数:     {s['n_over_25s']}")
    if s["n_over_25s"] > 0:
        w(f"  超 25 秒文件:     {fmt_list(s['over_25s_files'])}")
    status = "✅ 全部 ≤ 30 秒" if s["n_over_30s"] == 0 else f"❌ {s['n_over_30s']} 条超 30 秒"
    w(f"  结论:             {status}")
    w(f"  短于 3 秒条数:    {len(s['under_3s_files'])}")
    if s["under_3s_files"]:
        w(f"  短于 3 秒文件:    {fmt_list(s['under_3s_files'])}")
        w(f"  ⚠ 太短 (<3s) 的条目可能无足够上下文, 需检查")
    w(f"  【新标准对比】新标准推荐单条 5~15 秒")
    w()

    # ── 九、总时长评估 ──
    w("九、总时长评估")
    w("-" * 70)
    total_min = s["total_dur_min"]
    w(f"  总时长:           {total_min:.1f} 分钟")
    if total_min < 5:
        verdict = "❌ 不足 5 分钟, 仅够 smoke test"
    elif total_min < 15:
        verdict = "⚠ 不足 15 分钟, 勉强可用, 建议补充"
    elif total_min <= 30:
        verdict = "✅ 在推荐区间 15~30 分钟内"
    else:
        verdict = f"✅ 超过 30 分钟 ({total_min:.0f} 分钟), 数据充足"
    w(f"  评估:             {verdict}")
    w(f"  【新标准对比】推荐 15~30 分钟; 5 分钟仅够 smoke test")
    w()

    # ── 十、命名与配对 ──
    w("十、文件命名与两遍配对（旧标准 #4: 每条语料读2遍→2个文件）")
    w("-" * 70)
    groups = s.get("groups", {})
    if groups:
        w(f"  命名模式:         {s['naming_pattern']}  (X=组/遍号, Y=语料号)")
        for g in sorted(groups.keys()):
            nums = groups[g]
            w(f"  第 {g} 组:           {s.get(f'group_{g}_count', '?')} 条"
              f"  (编号 {s.get(f'group_{g}_range', '?')})")
        missing = s.get("missing_nums", [])
        w(f"  缺失编号:         {fmt_list(missing)}")
        for g in sorted(groups.keys()):
            only = s.get(f"group_{g}_only", [])
            if only:
                w(f"  仅第 {g} 组有的编号:  {fmt_list(only)}")
        if s.get("pairing_complete"):
            w(f"  配对完整性:       ✅ 两组编号完全一致, 两遍录制完整")
        else:
            w(f"  配对完整性:       ⚠ 存在缺失或单边, 请核实是否漏录")
    else:
        w(f"  命名模式:         无法识别 (非 'X - Y.wav' 格式)")
    w(f"  【新标准对比】新标准推荐命名: 音色ID_语料号_第n遍_会话ID.wav")
    w()

    # ── 十一、音量一致性 ──
    w("十一、音量一致性（旧标准 #7: 避免前后音量差异明显）")
    w("-" * 70)
    if s.get("n_silent", 0) > 0:
        w(f"  ⚠ 检测到 {s['n_silent']} 个静音/异常文件 (已排除): {fmt_list(s['silent_files'])}")
        w()
    w(f"  有效文件数:       {s.get('n_valid', s['n_ok'])} (排除静音后)")
    w(f"  RMS 均值:         {s['rms_mean']:.2f} dBFS")
    w(f"  RMS 标准差:       {s['rms_std']:.2f} dB")
    w(f"  RMS 范围:         {s['rms_min']:.2f} ~ {s['rms_max']:.2f} dBFS")
    w(f"  RMS 极差:         {s['rms_range']:.2f} dB")
    if s["rms_range"] < 3:
        vol = "✅ 音量一致 (极差 < 3dB)"
    elif s["rms_range"] < 6:
        vol = "⚠ 音量差异中等 (极差 3~6dB), 建议响度归一化"
    else:
        vol = f"❌ 音量差异大 (极差 {s['rms_range']:.1f}dB), 需响度归一化"
    w(f"  结论:             {vol}")
    w(f"  【新标准对比】新标准要求归一化到 -23 LUFS (需 ffmpeg loudnorm)")
    w()

    # ── 十二、需人工检查项 ──
    w("十二、需人工检查项（脚本无法自动判定）")
    w("-" * 70)
    w("  以下项目需要人工核实, 脚本无法从音频文件自动判断:")
    w()
    w("  □ 录音环境一致性  : 同一房间/麦克风/增益/时段 (统一性铁律)")
    w("  □ 混响 RT60      : < 0.4s, 需空录测量或听感判断 (新标准)")
    w("  □ 麦克风类型      : 大振膜电容麦 + 心型指向 (新标准)")
    w("  □ 疲劳管理        : 每 15~20 分钟休息 (旧标准 #5)")
    w("  □ 文本准确性      : 需 ASR + 人工校对, 100% 准确 (旧标准 #6)")
    w("  □ 文本归一化 TN   : 金额/卡号/日期/英文缩写读法 (银行场景)")
    w("  □ 语料覆盖        : 银行常用语料, 声韵母组合全 (旧标准 #6)")
    w("  □ 发音要求        : 面客语速/语调/音量 (旧标准 #7)")
    w("  □ 音色一致性      : 前后录音音色无明显差异 (旧标准 #7)")
    w("  □ 授权书签署      : 商用前置, 不签不录 (旧标准 #8)")
    w("  □ 元数据 JSON     : 新标准要求同名元数据文件")
    w()

    # ── 十三、综合评估 ──
    w("=" * 80)
    w("十三、综合评估")
    w("=" * 80)
    w()

    issues = []
    if not s["sr_all_48k"]:
        issues.append(("严重", "采样率非 48kHz", "需重录或重采样"))
    if not s["bit_all_16"]:
        issues.append(("中等", "位深非 16bit", "检查格式转换"))
    if s.get("n_silent", 0) > 0:
        issues.append(("严重", f"{s['n_silent']} 条静音/损坏文件", "需重录或删除"))
    if s["n_clipping"] > 0:
        issues.append(("严重", f"{s['n_clipping']} 条削波", "削波不可逆, 需重录"))
    if s["n_over_30s"] > 0:
        issues.append(("中等", f"{s['n_over_30s']} 条超 30 秒", "需在标点处切断"))
    if s["n_dc_fail"] > 0:
        issues.append(("轻微", f"{s['n_dc_fail']} 条 DC 偏移", "可用 ffmpeg 去直流"))
    if s["n_snr_fail"] > 0:
        issues.append(("中等", f"{s['n_snr_fail']} 条 SNR 不足 35dB", "检查录音环境"))
    if s["n_noise_fail"] > 0:
        issues.append(("中等", f"{s['n_noise_fail']} 条噪声 > -60dBFS", "检查录音环境"))
    n_sil = s["n_lead_fail"] + s["n_trail_fail"]
    if n_sil > 0:
        issues.append(("轻微", f"{n_sil} 条首尾静音不足 300ms", "需补静音或重录"))
    if s["rms_range"] >= 6:
        issues.append(("中等", f"音量极差 {s['rms_range']:.1f}dB", "需响度归一化"))
    elif s["rms_range"] >= 3:
        issues.append(("轻微", f"音量极差 {s['rms_range']:.1f}dB", "建议响度归一化"))
    if total_min < 5:
        issues.append(("严重", f"总时长仅 {total_min:.1f} 分钟", "数据不足, 需补录"))
    elif total_min < 15:
        issues.append(("轻微", f"总时长 {total_min:.1f} 分钟", "低于推荐 15 分钟"))
    if s["under_3s_files"]:
        issues.append(("轻微", f"{len(s['under_3s_files'])} 条短于 3 秒", "检查是否有效语料"))

    if not issues:
        w("  ✅ 自动检查项全部通过!")
        w("  数据集满足旧标准的技术要求, 可用于 TTS 微调。")
        w('  请继续完成上方"需人工检查项"后即可进入训练流程。')
    else:
        w("  发现以下问题 (按严重程度排序):")
        w()
        w(f"  {'标记':<4} {'严重度':<6} {'问题':<32} {'建议措施'}")
        w(f"  {'-'*4} {'-'*6} {'-'*32} {'-'*30}")
        for sev, prob, action in issues:
            marker = "❌" if sev == "严重" else "⚠"
            w(f"  {marker}  {sev:<6} {prob:<32} {action}")

    w()
    w("=" * 80)
    w("  报告结束 — 详细数据见同名 .json 文件")
    w("=" * 80)

    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Windows 终端默认 GBK, 切 UTF-8 以输出 emoji/特殊符号
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="TTS 微调数据集质量自动分析 (对照旧标准+新标准)"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=r"D:\招行\微调学习\客户录制数据集",
        help="数据集目录路径 (默认: 客户录制数据集)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"错误: 目录不存在: {data_dir}")

    print("=" * 60)
    print("  TTS 微调数据集质量自动分析")
    print(f"  数据集: {data_dir}")
    print("=" * 60)

    results, summary, err_results = analyze_dataset(data_dir)

    if err_results:
        print(f"\n⚠ {len(err_results)} 个文件分析失败:")
        for e in err_results:
            print(f"  {e['filename']}: {e.get('error', 'unknown')}")

    # 生成报告
    report = generate_report(results, summary, data_dir)

    # 保存文本报告
    txt_path = data_dir / "数据集质量分析报告.txt"
    txt_path.write_text(report, encoding="utf-8")

    # 保存 JSON 报告
    json_path = data_dir / "数据集质量分析报告.json"
    json_data = {
        "generated_at": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "standards_ref": {
            "old": "客户收到的8条录制要求",
            "new": "docs/03-数据集录制标准与方案.md",
        },
        "summary": {k: (int(v) if isinstance(v, (np.integer,)) else
                        float(v) if isinstance(v, (np.floating,)) else
                        v)
                     for k, v in summary.items()},
        "per_file": [{k: (int(v) if isinstance(v, (np.integer,)) else
                          float(v) if isinstance(v, (np.floating,)) else
                          v)
                       for k, v in r.items()}
                      for r in results],
    }
    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )

    # 打印报告到终端
    print()
    print(report)
    print(f"\n报告已保存: {txt_path}")
    print(f"详细数据: {json_path}")


if __name__ == "__main__":
    main()
