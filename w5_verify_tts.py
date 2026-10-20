"""W-5 TTS 前向数值验证（最小版，不依赖完整 Twinkle 训练循环）。

直接：HF 加载 Qwen3-TTS → 构造 batch → qwen3_tts_forward_step → Qwen3TTSLoss
→ 打印 talker/sub/total loss + shift_labels 开/关对照。

目的（W-5 最该先验的）：
  1. 选项A 编排（forward.py）+ loss（loss.py）在**真模型**上前向不崩、形状对、数值合理。
  2. shift_labels=False（修正 off-by-one）vs True（复现官方）的 loss 对照，确认修正方向。

不验（留后续）：
  - 选项A 子类 Qwen3TTSModel.forward 的 Twinkle forward_backward 集成（需完整训练循环组装，
    见 model.py docstring 的 TODO）。本脚本直接调 qwen3_tts_forward_step，等价验编排逻辑。
  - LoRA 挂载 + 训练降 loss（peft-lora-npu skill + 完整训练脚本，W-5 后续）。
  - 真实 mel 提取（ref_mels 用随机占位，speaker_encoder forward 不崩即可；真实 mel 留数据流水线）。

运行（容器内，deploy.sh 跑过后）：
  python w5_verify_tts.py --model_path /path/to/Qwen3-TTS-12Hz-1.7B-Base

⚠️ TODO（现场填）见各处注释。
"""

import argparse
import sys

import numpy as np
import torch

from tts.qwen3.data import Qwen3TTSDataConfig, build_qwen3_sample, collate_qwen3_batch
from tts.qwen3.forward import qwen3_tts_forward_step
from tts.qwen3.loss import Qwen3TTSLoss


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Qwen3-TTS-12Hz-1.7B-Base 本地权重路径")
    p.add_argument("--num_text_tokens", type=int, default=20,
                   help="随机 text_ids 长度（TODO: 真实用 tokenizer encode build_assistant_text(text)[:-5]）")
    p.add_argument("--num_codec_frames", type=int, default=64,
                   help="随机 audio_codes 帧数 C（TODO: 真实用官方 codec 编码真实音频）")
    p.add_argument("--device", default="npu", help="npu / cuda / cpu（TODO: NPU 用 'npu'）")
    return p.parse_args()


def _load_model(model_path: str, device: str):
    """加载 Qwen3-TTS 模型（用官方 qwen-tts 包，非 transformers AutoModel）。

    Qwen3-TTS 的 modeling 在 `qwen-tts` 包（不在 transformers 内置，config 也无 auto_map），
    故 transformers AutoModel 加载报 "does not recognize qwen3_tts"。正确加载：用 qwen_tts 包的
    `Qwen3TTSModel.from_pretrained`（高层封装），其 `.model` 属性是底层 `Qwen3TTSForConditionalGeneration`
    （有 `.talker`/`.code_predictor`/顶层 `.speaker_encoder`，正是 `qwen3_tts_forward_step` 要的）。

    环境约束（W-5 实测）：qwen-tts 钉 transformers==4.57.3，故 transformers 从 5.14.1 降到 4.57.3
    （twinkle 无版本约束，4.57.3 共存 OK）。昇腾无 flash-attn，attn_implementation='eager'。
    """
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,        # 昇腾原生 bf16（AGENTS.md 硬规则）
        attn_implementation="eager",       # 昇腾无 flash-attn（sft_12hz.py 用 flash_attention_2，改 eager）
    )
    model = qwen3tts.model                 # 底层 Qwen3TTSForConditionalGeneration
    model = model.to(device)
    model.eval()
    return model


def _build_random_batch(cfg: Qwen3TTSDataConfig, num_text: int, num_codec: int, device: str):
    """构造随机 batch 验前向（数值合理性，非真实数据）。

    TODO 现场替换为真实数据：
      - text_ids: tokenizer.encode(build_assistant_text(目标文本))[:-TEXT_IDS_TRIM]
      - audio_codes: 官方 codec 编码真实音频 → [C, 16]
      - ref_mels: 真实参考音频 mel（speaker_encoder 输入）
    """
    # 文本 token 在 Qwen3 文本词表段（~151xxx）；codec token 在 codec 段（~4xxx）
    text_ids = np.random.randint(1000, 150000, size=num_text).astype(np.int64)
    audio_codes = np.random.randint(0, 2048, size=(num_codec, cfg.num_code_groups)).astype(np.int64)  # < code_predictor vocab 2048 / talker vocab 3072

    sample = build_qwen3_sample(text_ids, audio_codes, cfg)
    # ref_mels: speaker_encoder 输入 mel。官方 extract_mels 产 [1, mel_frames, 128]（num_mels=128，
    # 非 Whisper 的 80），collate cat dim=0 → [batch, mel_frames, 128]。随机占位 [1, 128, 128]；TODO 真实 mel
    ref_mel = np.random.randn(1, 128, 128).astype(np.float32) * 0.1
    batch_np = collate_qwen3_batch([sample], cfg, ref_mels=[ref_mel])

    # np → torch + to device
    batch = {}
    for k, v in batch_np.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        if k == "ref_mels":
            t = t.to(torch.float32)          # speaker_encoder 内部转 dtype
        elif k in ("text_embedding_mask", "codec_embedding_mask", "codec_mask"):
            t = t.to(torch.bool)
        else:
            t = t.to(torch.long)
        batch[k] = t.to(device)
    return batch


def _run_once(model, batch, shift_labels: bool, device: str, label: str):
    """跑一次前向 + loss，打印数值。"""
    loss_fn = Qwen3TTSLoss(shift_labels=shift_labels)
    with torch.no_grad():
        outputs = qwen3_tts_forward_step(model, batch)
        result = loss_fn(inputs=None, outputs=outputs)
    talker_loss = float(outputs["talker_loss"].detach())
    sub_logits = outputs["sub_talker_logits"]
    sub_labels = outputs["sub_talker_labels"]
    total = float(result["loss"].detach())
    print(f"  [{label}] shift_labels={shift_labels}")
    print(f"      talker_loss      = {talker_loss:.4f}")
    print(f"      sub_talker_logits shape = {tuple(sub_logits.shape)}  (期望 [N, 15, V])")
    print(f"      sub_talker_labels shape = {tuple(sub_labels.shape)}  (期望 [N, 15])")
    print(f"      total loss       = {total:.4f}")
    return total


def main():
    args = _parse()
    print(f"device = {args.device}")
    if args.device == "npu":
        torch.npu.set_device(0)   # TODO: 多卡时按 ASCEND_RT_VISIBLE_DEVICES 已限定

    print(f"加载模型: {args.model_path}")
    model = _load_model(args.model_path, args.device)
    # 从真模型 config 读 num_code_groups（敲定 data.py 的强证据 16）
    ncb = getattr(model.config, "num_code_groups", 16)
    print(f"  model.config.num_code_groups = {ncb}  （data.py 默认 16，{'一致' if ncb == 16 else '需同步改 data.py NUM_CODE_GROUPS'}）")
    # cfg 从真模型 config 读 codec id（data.py 默认 4196 等是 W-1 调研错误，真实 config 是 2148 等；
    # 用默认值会让 codec_embedding 查表超词表报错）
    tc = model.config.talker_config
    cfg = Qwen3TTSDataConfig(
        num_code_groups=ncb,
        codec_pad_id=tc.codec_pad_id,
        codec_bos_id=tc.codec_bos_id,
        codec_eos_token_id=tc.codec_eos_token_id,
        codec_nothink_id=tc.codec_nothink_id,
        codec_think_bos_id=tc.codec_think_bos_id,
        codec_think_eos_id=tc.codec_think_eos_id,
    )

    print("构造随机 batch（TODO: 现场换真实 text_ids/audio_codes/ref_mels）...")
    batch = _build_random_batch(cfg, args.num_text_tokens, args.num_codec_frames, args.device)
    for k, v in batch.items():
        print(f"    {k}: {tuple(v.shape)} {v.dtype}")

    print("\n=== 前向 + loss 对照 ===")
    t_false = _run_once(model, batch, shift_labels=False, device=args.device, label="修正(无shift)")
    t_true = _run_once(model, batch, shift_labels=True, device=args.device, label="官方(含shift)")
    print(f"\n  对照: shift=False {t_false:.4f}  vs  shift=True {t_true:.4f}")
    print("  预期: shift=True(含 off-by-one) 的 sub_loss 偏差（cb15 被吞 + head 错配），")
    print("        shift=False(同位 per-head CE) 才是 depth-transformer next-token 正确语义。")
    print("        W-5 据此定 Qwen3TTSLoss.shift_labels 默认（当前默认 False）。")
    print("\n=== 验证完成 ===")
    print("下一步: 若数值合理 → 接 LoRA(add_adapter_to_model) + 完整 Twinkle 训练循环(Qwen3TTSModel) 做小步 SFT。")


if __name__ == "__main__":
    sys.exit(main() or 0)
