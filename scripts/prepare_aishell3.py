"""AISHELL-3 单说话人 → Qwen3-TTS manifest 预编码（数据流水线骨架）。

把开源 AISHELL-3（中文，16kHz）某说话人的录音，转成 Qwen3-TTS 训练可用的
manifest.jsonl（含预编码 audio_codes + 预提取 ref_mels）。产物对齐
`tts/common/multi_codebook.py` 的 manifest 字段约定 + `docs/05` §2/§3.3 的权威参数。

⚠️ 本脚本是 W-5 之后的「真实数据流水线」骨架，以下 API 本机无法验证
（qwen_tts / speech_tokenizer 包只在 NPU 容器），标 TODO，进容器对照
`docs/05` §2.1 / §3.3 + 官方 `finetuning/prepare_data.py` 补全：
  1. ✅ 已确认（probe 实跑）：`encode(audios, sr=None, return_dict=True)`，audios 接受
     numpy.ndarray（wav）+ sr=采样率；返回 .audio_codes → [T,16]。已填 encode_audio_codes。
  2. ✅ 已确认（probe）：qwen_tts 无官方 extract_mels，用 torchaudio kaldi.fbank fallback
     （ECAPA-TDNN 对 mel 格式敏感，fallback 格式一致性待 smoke-test 验证）。
  3. ✅ 已确认（容器下载后）：modelscope `cqliushuai/aishell3-dataset` 结构为
     `{root}/aishell3_16khz/{speaker_id}/*.wav`（无 .trn）+ 根目录 jsonl（文本在 audio 字段）。
     `load_aishell3_speaker` 已适配（读 jsonl 过滤 speaker_id）。T1/T2/T3 全部确认。

架构 gap（标注给训练侧）：`multi_codebook.py` 的 Processor 把 ref_audio 存路径，
但 `collate_qwen3_batch` 要 ref_mels 张量——中间「路径→mel」加载层本会话用「预提取
mel 存 .npy」绕开（manifest 存 mel 路径），训练时加载层留 TODO（见验证清单）。

运行（容器内，数据下载后——见 docs/10 步骤6 的 modelscope download 命令）：
  python scripts/prepare_aishell3.py \
    --aishell3_root /home/h00875519/aishell3 \
    --speaker_id SSB0005 \
    --text_jsonl aishell3_tts.jsonl \
    --output_dir data/qwen3_aishell3 \
    --speech_tokenizer_path /path/to/Qwen3-TTS-Tokenizer-12Hz
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

# ---- mel / 音频参数（docs/05 §3.3 权威 + 新会话必读）----
# speaker_encoder(ECAPA-TDNN) 输入：ref_mels [batch, mel_frames, mel_dim]
# 官方 extract_mels 产 [1, mel_frames, 128]（num_mels=128，非 Whisper 80），collate cat dim=0
MEL_DIM = 128          # mel_dim，docs/05 §3.3
MEL_HOP = 256          # hop_length（新会话必读 W-5 实测依据）
MEL_WIN = 1024         # win_length
MEL_FMAX = 12000.0     # fmax
TARGET_SR = 24000      # Qwen3-TTS ref 音频要求 24kHz（docs/05 §3.3 sample_rate=24000）
AISHELL_SR = 16000     # AISHELL-3 原始 16kHz → 需升采样到 24kHz
CODEC_SR = 24000       # speech_tokenizer 也吃 24kHz（与 mel 同源）


def resample_to_24k(wav: np.ndarray, sr_in: int) -> np.ndarray:
    """16kHz → 24kHz 升采样（Qwen3-TTS 要 24kHz）。

    TODO(容器探明)：优先用官方 prepare_data.py 的 resample 方式（一致性）；
    fallback 用 librosa/torchaudio。这里给 librosa 骨架。
    """
    import librosa  # 延迟导入，避免本机缺依赖时 import 失败
    if sr_in != TARGET_SR:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr_in, target_sr=TARGET_SR)
    return wav.astype(np.float32)


def extract_mel(wav_24k: np.ndarray) -> np.ndarray:
    """提取 mel 频谱 [mel_frames, MEL_DIM]，喂 speaker_encoder。

    TODO(容器探明)：官方有 extract_mels 函数（speaker_encoder 配套，格式/归一化一致
    是关键——ECAPA-TDNN 对 mel 格式敏感）。优先 import 官方 extract_mels；
    fallback 用 torchaudio MelSpectrogram 按上述参数。这里给 torchaudio 骨架。

    返回形状对齐 docs/05 §3.3 / w5_verify_tts.py 注释：[mel_frames, 128]（单条）。
    """
    import torch
    try:
        # TODO: from qwen_tts.... import extract_mels  # 容器探明路径
        # return extract_mels(wav_24k, sample_rate=TARGET_SR)  # 签名待确认
        raise ImportError('官方 extract_mels 路径未填，走 fallback')
    except ImportError:
        import torchaudio.compliance.kaldi as kaldi  # kaldi fbank 与 ECAPA-TDNN 常见配套
        t = torch.from_numpy(wav_24k).unsqueeze(0)   # [1, samples]
        mel = kaldi.fbank(
            t, num_mel_bins=MEL_DIM, frame_length=MEL_WIN / TARGET_SR * 1000,
            frame_shift=MEL_HOP / TARGET_SR * 1000, sample_frequency=TARGET_SR,
            high_freq=MEL_FMAX,
        )  # [mel_frames, 128]
        return mel.numpy()


def encode_audio_codes(wav_24k: np.ndarray, tokenizer) -> np.ndarray:
    """音频 → audio_codes [T, 16]（16 码本 RVQ，帧率 12Hz）。

    probe 确认签名：`encode(audios: Union[str, ndarray, List[str], List[ndarray]],
    sr: Optional[int]=None, return_dict=True)`。传 wav ndarray + sr=24kHz；
    返回 .audio_codes → [T,16]（docs/05 §2.1）。单条可能返 batch 维，squeeze 防御。
    """
    result = tokenizer.encode(wav_24k, sr=TARGET_SR)   # wav_24k 已升采样到 24kHz
    codes = result.audio_codes                          # [T, 16] 或 [1, T, 16]
    if hasattr(codes, 'cpu'):
        codes = codes.cpu().numpy()
    if codes.ndim == 3:                                 # encode 单条可能返 batch 维
        codes = codes[0]
    return codes.astype(np.int64)


def load_aishell3_speaker(root: Path, speaker_id: str, text_jsonl: str = 'aishell3_tts.jsonl'):
    """加载某说话人的 (wav路径, 文本) 列表。

    modelscope `cqliushuai/aishell3-dataset` 实测结构（容器下载后确认）：
      {root}/aishell3_16khz/{speaker_id}/{speaker_id}_xxxx.wav   (16kHz mono，无 .trn)
      {root}/{text_jsonl}  全量文本，每行 {"audio":"./aishell3_16khz/SSBxxxx/xxx.wav","text":"..."}
    文本靠 jsonl 的 audio 字段匹配 wav 文件名（SSBxxxx 目录下无独立文本文件）。
    """
    wav_dir = root / 'aishell3_16khz' / speaker_id
    jsonl_path = root / text_jsonl
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f'文本 jsonl 不存在: {jsonl_path}（需 modelscope download ... --include "{text_jsonl}"）')
    # 读 jsonl，按 audio 路径的倒数第二段 == speaker_id 过滤，建 wav文件名 → text
    text_map = {}
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            audio = r.get('audio', '').replace('\\', '/')
            parts = audio.split('/')
            # audio 形如 "./aishell3_16khz/SSB0005/SSB00050001.wav"，parts[-2]=SSB0005
            if len(parts) >= 2 and parts[-2] == speaker_id:
                text_map[parts[-1]] = r.get('text', '')
    pairs = []
    for wav_path in sorted(wav_dir.glob('*.wav')):
        text = text_map.get(wav_path.name)
        if not text:
            continue   # 该 wav 在 jsonl 无文本（可能在 val/test 拆分里），跳过
        pairs.append((str(wav_path), text))
    if not pairs:
        raise FileNotFoundError(
            f'在 {wav_dir} 下没找到有文本的 wav（jsonl={jsonl_path}，'
            f'检查 speaker_id={speaker_id} 或 text_jsonl 是否含该说话人）')
    return pairs


def read_wav(wav_path: str):
    """读 wav → (wav [samples], sr)。TODO: 用 torchaudio 或 soundfile。"""
    import soundfile as sf
    wav, sr = sf.read(wav_path, dtype='float32')
    if wav.ndim > 1:           # 转单声道
        wav = wav.mean(axis=1)
    return wav, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aishell3_root', required=True)
    ap.add_argument('--speaker_id', required=True, help='如 SSB0005')
    ap.add_argument('--text_jsonl', default='aishell3_tts.jsonl',
                    help='文本来源 jsonl（modelscope 数据集根目录下，默认全量 aishell3_tts.jsonl）')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--speech_tokenizer_path', required=True,
                    help='speech_tokenizer 权重路径（codec，独立模型，不训）——'
                         '传 {model_path}/speech_tokenizer 子目录（probe 确认权重在此）')
    ap.add_argument('--ref_audio', default=None,
                    help='参考音频路径（克隆条件，不传则取该说话人最长一条）')
    ap.add_argument('--dev_ratio', type=float, default=0.15,
                    help='dev 拆分比例（train≠dev，案例1 教训：文本不重叠）')
    ap.add_argument('--max_samples', type=int, default=0,
                    help='smoke-test 用：只处理前 N 条（0=全量）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    mel_dir = out / 'ref_mels'
    out.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(exist_ok=True)

    # ---- 1. 加载 codec（Qwen3TTSTokenizer，独立模型，不训，docs/05 §3.4）----
    print(f'加载 speech tokenizer: {args.speech_tokenizer_path}')
    # probe 确认：Qwen3TTSTokenizer 在 modeling_qwen3_tts（非 speech_tokenizer 模块名）；
    # 权重在 {model}/speech_tokenizer/ 子目录，--speech_tokenizer_path 传该子目录
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTokenizer
    tokenizer = Qwen3TTSTokenizer.from_pretrained(args.speech_tokenizer_path)

    # ---- 2. 加载说话人音频 + 文本 ----
    pairs = load_aishell3_speaker(Path(args.aishell3_root), args.speaker_id, args.text_jsonl)
    print(f'说话人 {args.speaker_id}: {len(pairs)} 条')
    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]
        print(f'  smoke-test: 只取前 {len(pairs)} 条')

    # ---- 3. 选 ref_audio 并预提取其 mel（克隆条件，所有样本共用同一 ref）----
    # 注意：ref_audio 与目标音频可不同（零样本克隆语义）；这里用同一说话人的一条做 ref
    ref_path = args.ref_audio or max(pairs, key=lambda p: len(read_wav(p[0])[0]))[0]
    print(f'ref_audio = {ref_path}')
    ref_wav, ref_sr = read_wav(ref_path)
    ref_wav_24k = resample_to_24k(ref_wav, ref_sr)
    ref_mel = extract_mel(ref_wav_24k)                    # [mel_frames, 128]
    ref_mel_path = mel_dir / 'ref_mel.npy'
    np.save(ref_mel_path, ref_mel)
    print(f'  ref_mel {ref_mel.shape} → {ref_mel_path}')

    # ---- 4. train/dev 拆分（train≠dev，文本不重叠，案例1 根因规避）----
    random.shuffle(pairs)
    n_dev = max(1, int(len(pairs) * args.dev_ratio))
    dev = pairs[:n_dev]
    train = pairs[n_dev:]
    # 文本去重保 train 独占（dev 文本不出现在 train）
    train_texts = {t for _, t in train}
    dev = [(w, t) for w, t in dev if t not in train_texts] or dev
    print(f'  train {len(train)} / dev {len(dev)}')

    # ---- 5. 逐条预编码：升采样 → mel(目标用于 ref 比对，可选) → audio_codes ----
    def build_rows(split_pairs):
        rows = []
        for wav_path, text in split_pairs:
            wav, sr = read_wav(wav_path)
            wav_24k = resample_to_24k(wav, sr)
            codes = encode_audio_codes(wav_24k, tokenizer)   # [T, 16]
            rows.append({
                'audio': wav_path,
                'text': text,
                'language': 'Chinese',
                'ref_audio': ref_path,
                'ref_mel_path': str(ref_mel_path),            # 预提取 mel，绕开训练时加载 gap
                'audio_codes': codes.tolist(),               # docs/05 §2.1: .cpu().tolist()
            })
        return rows

    for name, split_pairs in [('train', train), ('dev', dev)]:
        rows = build_rows(split_pairs)
        out_jsonl = out / f'{name}.jsonl'
        with open(out_jsonl, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'  {out_jsonl}: {len(rows)} 行')

    print('\n完成。下一步：')
    print('  1. 容器内填 TODO（Qwen3TTSTokenizer.encode 签名 + extract_mels 路径）后重跑')
    print('  2. 训练时 ref_mels 加载层（.npy → 张量，喂 collate_qwen3_batch）待实现')
    print('  3. 跑 scripts/train_qwen3_tts_lora.py --train_jsonl ... --dev_jsonl ...')


if __name__ == '__main__':
    main()
