"""Qwen3-TTS LoRA SFT 训练脚本（Twinkle，单卡 NPU，骨架）。

接 W-5 之后：把已验证的组件（Qwen3TTSModel / qwen3_tts_forward_step / Qwen3TTSLoss /
build_qwen3_sample / collate_qwen3_batch）装成可跑 LoRA SFT 的训练循环。

数据侧（smoke-test 简化路径）：
  - 自定义 torch Dataset 读 manifest（prepare_aishell3.py 产出），直接调
    build_qwen3_sample + np.load(ref_mel_path)，绕过 Twinkle Dataset/Processor/Template。
  - 原因：W-3 组件链（Processor→Template→InputProcessor）里「ref_audio 路径→ref_mels
    张量」的加载层未实现（template.py:60 注释「mel 提取留 W-5」），走标准链路会缺
    ref_mels。本脚本在 Dataset.__getitem__ 内联加载，先跑通流程。
  - TODO（迁回 Twinkle Dataset）：在 Qwen3TTSTemplate._encode_messages 读 ref_mel_path
    → np.load → sample['ref_mels']，并把 ref_mels 加进 Qwen3TTSInputProcessor 白名单 +
    padding_map，即可回到 `Dataset.map(Qwen3TTSProcessor).encode()` 标准范式。

训练循环（保留 Twinkle 抽象）：
  Qwen3TTSModel(model_cls=...) → kernelize → add_adapter_to_model(LoRA, alpha=2r)
  → set_optimizer(AdamW) → set_lr_scheduler(cosine) → set_loss(Qwen3TTSLoss)
  → forward_backward + clip_grad_and_step + save。

⚠️ 本机无法实测（qwen_tts/twinkle/torch_npu 只在 NPU 容器），标 TODO，进容器点亮。
运行（容器内，卡7）：
  ASCEND_RT_VISIBLE_DEVICES=7 python scripts/train_qwen3_tts_lora.py \
    --model_path /home/h00875519/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl data/qwen3_aishell3/train.jsonl \
    --dev_jsonl   data/qwen3_aishell3/dev.jsonl \
    --output_dir  outputs/qwen3_tts_lora
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, get_logger
from twinkle.kernel import kernelize

from tts.qwen3.data import (
    Qwen3TTSDataConfig, build_assistant_text, build_qwen3_sample, collate_qwen3_batch,
)
from tts.qwen3.loss import Qwen3TTSLoss
from tts.qwen3.model import Qwen3TTSModel

logger = get_logger()


# ---------------- 数据侧（smoke-test 简化）----------------

class TTSManifestDataset(TorchDataset):
    """读 manifest.jsonl → build_qwen3_sample + np.load(ref_mel)。

    复用 W-2 的 build_qwen3_sample（双通道布局）+ collate_qwen3_batch（padding），
    跳过 Processor/Template/Dataset 抽象以绕开 ref_mels 注入 gap。
    """

    def __init__(self, jsonl_path: str, cfg: Qwen3TTSDataConfig, tokenizer):
        self.rows = [json.loads(line) for line in open(jsonl_path, encoding='utf-8')]
        self.cfg = cfg
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # 文本编码：build_assistant_text 包成 assistant 轮 → encode → 去尾部 TEXT_IDS_TRIM
        # （镜像 Qwen3TTSTemplate._encode_messages 的逻辑，TEXT_IDS_TRIM=5）
        from tts.qwen3.template import TEXT_IDS_TRIM
        text_ids = np.array(
            self.tokenizer.encode(build_assistant_text(r['text'])), dtype=np.int64,
        )[:-TEXT_IDS_TRIM]
        audio_codes = np.array(r['audio_codes'], dtype=np.int64)   # [C, 16]
        sample = build_qwen3_sample(text_ids, audio_codes, self.cfg)
        ref_mel = np.load(r['ref_mel_path'])                        # [mel_frames, 128]
        return sample, ref_mel


def make_collate_fn(cfg: Qwen3TTSDataConfig):
    def collate(batch):
        samples = [b[0] for b in batch]
        ref_mels = [b[1] for b in batch]
        return collate_qwen3_batch(samples, cfg, ref_mels=ref_mels)
    return collate


def batch_to_device(batch: dict, device: str) -> dict:
    """np → torch + to device（镜像 w5_verify_tts.py 的转换）。"""
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        if k == 'ref_mels':
            t = t.to(torch.float32)          # speaker_encoder 内部转 dtype
        elif k in ('text_embedding_mask', 'codec_embedding_mask', 'codec_mask'):
            t = t.to(torch.bool)
        else:
            t = t.to(torch.long)
        out[k] = t.to(device)
    return out


# ---------------- 模型加载 ----------------

def load_model(args, device: str) -> Qwen3TTSModel:
    """加载 Qwen3-TTS 为 Twinkle Qwen3TTSModel（选项A 子类）。

    ⚠️ 加载 TODO（容器实测，核心难题）：
    TransformersModel.__init__ 用 `AutoConfig.from_pretrained(model_id, trust_remote_code=True)`
    读 config（transformers.py:283）。但 Qwen3-TTS 的 Qwen3TTSConfig 在 qwen_tts 包、
    config 无 auto_map，AutoConfig 可能不认 qwen3_tts → 缺 talker_config 等。
    解法二选一（容器验证）：
      A. 传预构造的 Qwen3TTSConfig（用 qwen_tts 加载 config）作 `config=` 参数：
         `Qwen3TTSModel(model_id=path, model_cls=Qwen3TTSForConditionalGeneration,
                        config=qwen3tts_config, mixed_precision='bf16',
                        attn_implementation='eager')`
      B. override Qwen3TTSModel.__init__，用 qwen_tts.Qwen3TTSModel.from_pretrained
         加载高层 → 取 .model 作底层，再补 super().__init__ 的策略/optimizer_group 初始化。
    下面给 A 作起点，容器内若 AutoConfig 失败再改 B。
    """
    # probe 确认：Qwen3TTSForConditionalGeneration 在 modeling_qwen3_tts；
    # Qwen3TTSConfig 在 configuration_qwen3_tts。AutoConfig 不认 qwen3_tts（probe L2 实测），
    # 须显式传 config=（方案A），否则 TransformersModel.__init__ 的 AutoConfig 会报错。
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    hf_config = Qwen3TTSConfig.from_pretrained(args.model_path)
    model = Qwen3TTSModel(
        model_id=args.model_path,
        model_cls=Qwen3TTSForConditionalGeneration,
        config=hf_config,                      # 显式传（AutoConfig 不认 qwen3_tts）
        mixed_precision='bf16',
        attn_implementation='eager',          # 昇腾无 flash-attn（sft_12hz.py 的 flash_attention_2 改 eager）
    )
    return model


def build_data_config(model) -> Qwen3TTSDataConfig:
    """从真模型 config 读 codec id（w5_verify_tts.py 实测做法，避免默认值错误）。"""
    tc = model.model.config.talker_config
    ncb = getattr(model.model.config, 'num_code_groups', 16)
    return Qwen3TTSDataConfig(
        num_code_groups=ncb,
        codec_pad_id=tc.codec_pad_id,
        codec_bos_id=tc.codec_bos_id,
        codec_eos_token_id=tc.codec_eos_token_id,
        codec_nothink_id=tc.codec_nothink_id,
        codec_think_bos_id=tc.codec_think_bos_id,
        codec_think_eos_id=tc.codec_think_eos_id,
    )


# ---------------- 训练 ----------------

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--train_jsonl', required=True)
    ap.add_argument('--dev_jsonl', default=None)
    ap.add_argument('--output_dir', required=True)
    # LoRA 超参（skill 第5节基线：alpha=2r，勿固定 alpha=16 的 case1 坑）
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--target_modules', default='all-linear')
    # 优化器/调度（skill：lr1e-4/wd0.1/betas(0.9,0.95)/cosine+warmup0.03；epochs3起）
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--warmup_ratio', type=float, default=0.03)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--grad_accum', type=int, default=16)     # 单卡补 batch
    ap.add_argument('--max_steps', type=int, default=0, help='smoke-test 用，0=按 epochs')
    ap.add_argument('--save_interval', type=int, default=50)
    ap.add_argument('--log_interval', type=int, default=10)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Twinkle 单卡 NPU 初始化 ----
    device_mesh = DeviceMesh.from_sizes(fsdp_size=1, dp_size=1)   # 单卡偏好（AGENTS.md）
    twinkle.initialize(mode='npu', global_device_mesh=device_mesh)
    device = 'npu'
    torch.npu.set_device(0)   # ASCEND_RT_VISIBLE_DEVICES=7 已限定卡

    # ---- 2. 加载模型 + kernelize（NPU 内核融合 CANN+FLA+fused-CE）----
    logger.info(f'加载模型: {args.model_path}')
    model = load_model(args, device)
    _torch_baseline = __import__('os').environ.get('TWINKLE_TORCH_BASELINE', '').lower() in ('1', 'true', 'yes')
    if not _torch_baseline:
        model = kernelize(model)   # FLA fast path 可能 fallback torch（非阻塞，见新会话必读待解项）

    # ---- 3. LoRA（alpha=2r，target_modules=all-linear 覆盖 talker+code_predictor）----
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,          # alpha=2r（peft-lora-npu skill；baseline 的 4r 是 fsdp2 默认勿沿用）
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias='none',
    )
    model.add_adapter_to_model('default', lora_config,
                               gradient_accumulation_steps=args.grad_accum)

    # ---- 4. 优化器（AdamW + wd0.1 防过拟合，skill TTS 经验）+ 调度 + loss ----
    # 注：fsdp2 baseline 用 Muon（已验 NPU 可用）；本脚本按 skill 第5节 TTS 基线用 AdamW
    #（betas/wd 对 TTS 过拟合控制更直接）。若 NPU 上 AdamW 有问题可换 Muon 对照。
    model.set_optimizer(
        optimizer_cls='AdamW',
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    # 文本 tokenizer 用于 Dataset 内编码 build_assistant_text(text)（probe L3 确认：
    # Qwen3-TTS 文本侧是标准 Qwen3 文本 tokenizer，模型目录有 tokenizer_config.json，
    # AutoTokenizer 能加载；model.tokenizer 可能为 None，故直接 AutoTokenizer）
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Qwen3TTSDataConfig 的 codec id 从 model.model.config.talker_config 读（w5_verify 做法）
    cfg = build_data_config(model)

    # 数据集 + 步数
    train_ds = TTSManifestDataset(args.train_jsonl, cfg, tokenizer) if cfg else None
    steps_per_epoch = len(train_ds) // args.batch_size if train_ds else 0
    total_steps = args.max_steps or (steps_per_epoch * args.epochs)
    num_warmup = int(args.warmup_ratio * total_steps) if total_steps else 0
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',   # probe L4 确认：在 twinkle.module.scheduler，construct_class 能解析
        num_warmup_steps=num_warmup,
        num_training_steps=total_steps,
    )
    model.set_loss(Qwen3TTSLoss(shift_labels=False), adapter_name='default')   # shift=False 修正 off-by-one

    if train_ds is None:
        logger.error('cfg/tokenizer 未就绪（TODO 未点亮），无法建数据集；先点亮容器 TODO')
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate_fn(cfg), num_workers=2,
    )

    # ---- 5. 训练循环 ----
    optimizer_group = model.optimizer_group['default']
    logger.info(f'Total steps: {total_steps} (epochs={args.epochs}, steps/epoch={steps_per_epoch})')
    step = 0
    for epoch in range(args.epochs):
        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, device)
            # ⚠️ TODO(容器实测，model.py forward override 关键点)：
            #   model.forward 调 optimizer_config.processor(inputs)（model.py:74）。
            #   本脚本传的是已 collate 的 2D batch dict（有 input_ids），_not_encoded=False
            #   跳过 template.batch_encode；但 default InputProcessor 对「已 collate 2D batch」
            #   的行为未验（可能 to_transformers_dict 转换 OK，也可能报错）。
            #   若报错：set processor=Qwen3TTSInputProcessor，或 model.py 对已 collate batch 跳过 processor。
            model.forward_backward(inputs=batch)          # task 不传（override pop）
            model.clip_grad_and_step()
            step += 1
            if step % args.log_interval == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(f'epoch {epoch} step {step}/{total_steps} metric: {metric}')
            if step % args.save_interval == 0:
                model.save(f'checkpoint-{step}', output_dir=str(out),
                           adapter_name='default', save_optimizer=True,
                           consumed_train_samples=step * args.batch_size)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    model.save('last-checkpoint', output_dir=str(out), adapter_name='default',
               save_optimizer=True, consumed_train_samples=step * args.batch_size)
    logger.info(f'训练结束，adapter 存于 {out}')

    # eval 留 WER/SIM 阶段：model.py 未 override forward_only（TODO #2），
    #   走基类 forward_only 会单次 self.model(**inputs) 对 Qwen3-TTS 崩，故本脚本只盯 train loss。


if __name__ == '__main__':
    train()
