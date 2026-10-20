#!/bin/bash
# w5_deploy.sh —— 在容器内应用本项目 tts/ 代码 + 跑回归单测（W-5 实操依据）
#
# 职责（只管"应用本项目的修改 + 验证"，环境你自己装）：
#   段0  前置检查（twinkle/transformers/torch_npu/peft 已装？没装就报错退出，不替你装）
#   段1  铺 tts/ 全部 .py（heredoc，幂等，可重跑）
#   段2  跑 4 个回归单测（W-1~W-4 的 CPU 成果在容器内回归）
#   段3  选项A 子类 Qwen3TTSModel import 验证
#
# 前置（你已做，见 docs/07 §5.1-5.2 + 会话9 qwen-tts 装好）：
#   git clone https://github.com/modelscope/twinkle /workspace/twinkle
#   cd /workspace/twinkle && git checkout 78d9304 && pip install -e .
#   pip install peft flash-linear-attention
#   pip install -U qwen-tts        # 拉全依赖：transformers==4.57.3 + accelerate==1.12.0 + sox + torchaudio + librosa ...
#   # 注：qwen-tts 钉 transformers==4.57.3，故从 5.14.1 降到 4.57.3（twinkle 无版本约束，4.57.3 共存 OK，会话9 实测）
#
# 用法：bash w5_deploy.sh [/workspace]   （默认装到 /workspace）
set -euo pipefail

INSTALL_DIR="${1:-/workspace}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"
export PYTHONPATH="$INSTALL_DIR:${PYTHONPATH:-}"

echo "================================================"
echo " w5_deploy.sh  ->  $INSTALL_DIR"
echo "================================================"

# ---- 段0 前置环境检查 ----
echo "[0/3] 前置环境检查..."
python - <<'PYCHECK'
import sys
ok = True
for m in ("torch", "torch_npu", "twinkle", "transformers", "peft", "qwen_tts"):
    try:
        __import__(m)
        print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
        ok = False
if not ok:
    print("\n  -> 环境未就绪，先按 docs/07 §5.1-5.2 + 装 qwen-tts 再跑本脚本")
    sys.exit(1)
import transformers
print(f"  OK   transformers {transformers.__version__}")
print("  环境就绪")
PYCHECK

# ---- 段1 铺 tts/ + scripts/ 代码（heredoc，幂等）----
echo "[1/3] 铺 tts/ + scripts/ 代码..."

mkdir -p "$(dirname "tts/__init__.py")"
cat > "tts/__init__.py" <<'PYEOF'
"""本项目 TTS 训练组件（开发期；成果稳定后回流 twinkle/src/twinkle/）。

结构（通用框架层 + 模型 cookbook）：
- `tts/common/`：模型无关的通用基类（PR 回流 twinkle 的目标）
    - `MultiCodebookPreprocessor`：manifest 行 → Trajectory（audio_codes 走 user_data）
    - `MultiCodebookLoss`：多码本 next-token CE 契约基类（W-4 填）
- `tts/qwen3/`：Qwen3-TTS cookbook（特有）
    - `data.py`：双通道 InputFeature 构建（移植官方 collate_fn，已单测）
    - `processor.py`：Qwen3TTSProcessor（用通用基类）
    - `template.py`：Qwen3TTSTemplate（override _encode_messages，W-3）
    - `forward.py`：qwen3_tts_forward_step（官方 5 步前向编排）
    - `loss.py`：Qwen3TTSLoss（2 段 CE，W-4）

架构见 `docs/05-W1-多码本组织调研.md`；设计见 `docs/06-W2-组件设计.md`。
"""

from .common import MultiCodebookLoss, MultiCodebookPreprocessor

__all__ = ["MultiCodebookPreprocessor", "MultiCodebookLoss"]
PYEOF
mkdir -p "$(dirname "tts/common/__init__.py")"
cat > "tts/common/__init__.py" <<'PYEOF'
"""通用多码本 TTS 组件（框架层目标，成果稳定后回流 `twinkle/src/twinkle/`）。

这里放**模型无关**的抽象：
- `MultiCodebookPreprocessor`：manifest 行 → Trajectory（audio_codes 走 user_data）
- `MultiCodebookLoss`：多码本 next-token CE 的契约基类

具体模型（Qwen3 / MOSS / CosyVoice）在各自 cookbook 子目录里子类化实现。
maintainer 确认：audio_codes 放 `Trajectory.user_data`，不改 Message（见 docs/02 §4.3）。
"""

from .multi_codebook import MultiCodebookPreprocessor
from .multi_codebook_loss import MultiCodebookLoss

__all__ = ["MultiCodebookPreprocessor", "MultiCodebookLoss"]
PYEOF
mkdir -p "$(dirname "tts/common/multi_codebook.py")"
cat > "tts/common/multi_codebook.py" <<'PYEOF'
"""通用多码本 TTS Preprocessor 基类（模型无关）。

把 manifest 行打包成 Twinkle `Trajectory`：text 放 `messages`，audio_codes / ref_audio /
language 放 `user_data`（PyArrow-stable，用 `pack_value` 编码）。下游自定义 Template
（如 `tts.qwen3.template.Qwen3TTSTemplate`）从 `user_data` 取出 audio_codes 构建具体
InputFeature。

manifest 字段名用类属性约定，子类可覆盖适配不同数据集（Qwen3 用默认即可）。
"""

from typing import Any, Dict

from twinkle.data_format import Message, Trajectory, pack_value
from twinkle.preprocessor import Preprocessor


class MultiCodebookPreprocessor(Preprocessor):
    """manifest 行 → Trajectory（多码本 audio_codes 走 user_data）。

    子类一般只需覆盖字段名类属性（或直接用默认），复杂定制可覆盖 `preprocess`。
    """

    # manifest 字段名约定（子类可覆盖）
    TEXT_FIELD = "text"
    AUDIO_CODES_FIELD = "audio_codes"          # [[...], shape C×N]
    REF_AUDIO_FIELD = "ref_audio"              # 参考音频路径（或预编码 codes，视模型而定）
    LANGUAGE_FIELD = "language"
    DEFAULT_LANGUAGE = "Auto"

    def __call__(self, rows: Dict[str, list]) -> Dict[str, list]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(r) for r in rows]
        return self.map_row_to_col(rows)

    def preprocess(self, row) -> Trajectory:
        text = row[self.TEXT_FIELD]
        audio_codes = row[self.AUDIO_CODES_FIELD]
        ref_audio = row.get(self.REF_AUDIO_FIELD)
        language = row.get(self.LANGUAGE_FIELD, self.DEFAULT_LANGUAGE)
        return Trajectory(
            messages=[Message(role="user", content=text)],
            user_data=[
                ("audio_codes", pack_value(audio_codes)),
                ("ref_audio", pack_value(ref_audio)),
                ("language", pack_value(language)),
            ],
        )
PYEOF
mkdir -p "$(dirname "tts/common/multi_codebook_loss.py")"
cat > "tts/common/multi_codebook_loss.py" <<'PYEOF'
"""通用多码本 Loss 基类（模型无关，W-4 填实现）。

契约（子类按模型实现 `__call__`）：
- 多码本 next-token CE：每个码本一路 CE，按 `codebook_weights` 加权求和。
- `ar_codebook`：哪个码本由 AR 主模型预测（Qwen3=码本0 由 talker 算；其余由 depth
  transformer 算）——该路由决定是否走标准 shift+CE，残差码本走各自模型的前向。
- 设 `require_logps=False, require_logits=True`：跳过 Twinkle 单码本
  `selective_log_softmax`（`utils/torch_utils.py:62`），保留 `outputs['logits']`
  （`model/transformers/transformers.py:592/617`），由子类自算多码本 CE。

具体模型子类：
- Qwen3 2 段式（talker CE + 0.3×code_predictor CE）：见 `tts.qwen3.loss.Qwen3TTSLoss`
- MOSS 时间同步 / CosyVoice：各自 cookbook 子类（W-6）

⚠️ Qwen3 off-by-one（W-4 实测确认，见 `tts/qwen3/loss.py` + `tts/qwen3/tests/test_loss.py`）：
官方 code_predictor 的 forward_finetune 对 [N,15,V] logits 调继承的 `ForCausalLMLoss`，后者在
最后一轴做标准 LM shift → 对码本轴错位（head 预测码本 j+1 却对码本 j+2 算 loss，且 cb15 被末位
-100 pad 吞掉不算）。同位 per-head CE（无 shift）才是 depth-transformer next-token 的正确语义。
子类约定：code_predictor 段用 `shift_labels` 开关——False（默认）= 同位 per-head CE（修正），
True = 复现官方含 shift（W-5 NPU 实测对照用）。talker 段（标准时间轴 [B,T,V]）shift 正确，透传。
"""

from typing import Optional, Sequence

from twinkle.data_format.output import LossOutput
from twinkle.loss.base import Loss


class MultiCodebookLoss(Loss):
    """多码本 next-token CE 基类。

    Args:
        codebook_weights: 各码本 CE 权重，None=等权（Qwen3 的残差段权重 0.3 在子类里体现）。
        ar_codebook: AR 主码本 index（该码本由主模型 next-token 预测，走标准 shift+CE）。
    """

    require_logps = False
    require_logits = True

    def __init__(
        self,
        codebook_weights: Optional[Sequence[float]] = None,
        ar_codebook: int = 0,
    ):
        self.codebook_weights = codebook_weights
        self.ar_codebook = ar_codebook

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        raise NotImplementedError(
            "子类实现：按模型多码本组织算 CE。Qwen3 见 tts.qwen3.loss.Qwen3TTSLoss。"
        )
PYEOF
mkdir -p "$(dirname "tts/qwen3/__init__.py")"
cat > "tts/qwen3/__init__.py" <<'PYEOF'
"""Qwen3-TTS cookbook（特有部分）。

复用通用基类 `tts.common`，实现 Qwen3-TTS 特定的：
- 数据构建（双通道 InputFeature）：`data.py` 的 `build_qwen3_sample` + `collate_qwen3_batch`
- Processor：`Qwen3TTSProcessor`（用通用基类，Qwen3 默认字段名）
- Template：`Qwen3TTSTemplate`（override _encode_messages，W-3）
- 前向编排：`forward.py` 的 `qwen3_tts_forward_step`（官方 5 步）
- Loss：`Qwen3TTSLoss`（2 段 CE，W-4）

架构见 `docs/05-W1-多码本组织调研.md`；设计见 `docs/06-W2-组件设计.md`。
"""

from .data import Qwen3TTSDataConfig, build_qwen3_sample, collate_qwen3_batch
from .input_processor import Qwen3TTSInputProcessor
from .processor import Qwen3TTSProcessor

__all__ = [
    "Qwen3TTSDataConfig",
    "build_qwen3_sample",
    "collate_qwen3_batch",
    "Qwen3TTSProcessor",
    "Qwen3TTSInputProcessor",
]
PYEOF
mkdir -p "$(dirname "tts/qwen3/data.py")"
cat > "tts/qwen3/data.py" <<'PYEOF'
"""Qwen3-TTS 多码本训练样本构建器（W-2/W-3）。

忠实移植自官方 `Qwen3-TTS/finetuning/dataset.py` 的 `TTSDataset.__getitem__` + `collate_fn`
（见 `docs/05-W1-多码本组织调研.md` 第 4 节）。把 manifest 一行（text_ids + audio_codes + ref_mel）
组织成 Talker/Code predictor 可 forward 的双通道 InputFeature：

  input_ids      [T, 2]   双通道：[:,0]=文本 token，[:,1]=codec token（码本0）
  codec_ids      [T, 16]  全码本（喂 Code predictor）
  codec_0_labels [T]      码本0 的 label，非目标帧置 -100
  text_embedding_mask  [T]   文本嵌入有效位
  codec_embedding_mask [T]   codec 嵌入有效位（pos6 置 False，留给 speaker emb）
  codec_mask           [T]   哪些是目标音频帧（取 hidden/codec_ids 用，不含末尾 eos）
  attention_mask       [T]   因果注意力有效位

序列布局（单样本，T = 8 + L + C，L=text_ids_len，C=codec_ids_len）：
  pos [0:3]            : text_ids[:3]           (文本通道) / 0 (codec 通道)
  pos [3:8]            : tts_pad (文本) / [nothink, think_bos, think_eos, 0(spk槽), codec_pad] (codec)
  pos [8 : 8+L-3]      : text_ids[3:]           / codec_pad
  pos [8+L-3]          : tts_eos                / codec_pad
  pos [8+L-2]          : tts_pad                / codec_bos
  pos [8+L-1 : 8+L-1+C]: tts_pad                / audio_codec_0 (码本0 token)   ← 目标音频帧
  pos [8+L-1+C] = T-1  : tts_pad                / codec_eos

speaker_emb 在训练步内注入 codec 嵌入张量的 pos6（见 forward 编排，本模块不涉及）。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

NUM_CODE_GROUPS = 16  # 12Hz 模型实际值（见 docs/05 §2.2，待 config.json 最终敲定）


@dataclass
class Qwen3TTSDataConfig:
    """Qwen3-TTS 双通道布局所需的特殊 token id（来自 Qwen3TTSConfig.talker_config）。

    ⚠️ W-5 实测修正（会话9）：原默认值 4196 等是 W-1 调研错误，真实 config 的 codec id
    在 2xxx 段（见 docs/05 §4）。训练/验证时建议从 `model.config.talker_config` 读真实值
    构造本类（如 w5_verify_tts.py 所做），而非依赖默认值。默认值已修正为真实 config 值。
    """

    # 文本通道（Qwen3 文本词表段，151xxx，config 顶层 tts_*_token_id）
    tts_pad_token_id: int = 151671
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    # codec 通道（codec 词表段，2xxx，config.talker_config.codec_*_id）
    codec_pad_id: int = 2148
    codec_bos_id: int = 2149
    codec_eos_token_id: int = 2150
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157
    num_code_groups: int = NUM_CODE_GROUPS


def build_assistant_text(text: str) -> str:
    """官方 `_build_assistant_text`：把目标文本包成 assistant 轮（ChatML）。

    真实 text_ids 由 Qwen3 文本 tokenizer 对该串编码、再 `[:, :-5]` 得到。
    本模块只负责布局，text_ids 由上游（Preprocessor/Template）传入。
    """
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_qwen3_sample(
    text_ids: np.ndarray,
    audio_codes: np.ndarray,
    cfg: Qwen3TTSDataConfig,
) -> Dict[str, np.ndarray]:
    """把单条 (text_ids, audio_codes) 组成双通道 InputFeature（未 pad）。

    Args:
        text_ids: 1D int array [L]，L>=4（官方 `[:, :-5]` 后的文本 token，含 assistant 标签）。
        audio_codes: 2D int array [C, num_code_groups]，目标音频多码本。
        cfg: 特殊 token id。

    Returns:
        dict，每项是长度 T=8+L+C 的 numpy 数组（input_ids/codec_ids 是 2D）。
    """
    text_ids = np.asarray(text_ids).reshape(-1)
    audio_codes = np.asarray(audio_codes)
    assert audio_codes.ndim == 2 and audio_codes.shape[1] == cfg.num_code_groups, (
        f"audio_codes shape 应为 [C, {cfg.num_code_groups}]，实得 {audio_codes.shape}"
    )
    L = int(text_ids.shape[0])
    C = int(audio_codes.shape[0])
    assert L >= 4, f"text_ids_len L={L} 过短（需 >=4 以满足 [3:] 非空）"

    T = 8 + L + C
    audio_codec_0 = audio_codes[:, 0]  # [C]
    ncb = cfg.num_code_groups

    input_ids = np.zeros((T, 2), dtype=np.int64)
    codec_ids = np.zeros((T, ncb), dtype=np.int64)
    codec_0_labels = np.full((T,), -100, dtype=np.int64)
    text_embedding_mask = np.zeros((T,), dtype=bool)
    codec_embedding_mask = np.zeros((T,), dtype=bool)
    codec_mask = np.zeros((T,), dtype=bool)
    attention_mask = np.zeros((T,), dtype=np.int64)

    # ---- 文本通道 input_ids[:,0] ----
    input_ids[0:3, 0] = text_ids[0:3]
    input_ids[3:7, 0] = cfg.tts_pad_token_id
    input_ids[7, 0] = cfg.tts_bos_token_id
    input_ids[8 : 8 + L - 3, 0] = text_ids[3:]          # L-3 个
    input_ids[8 + L - 3, 0] = cfg.tts_eos_token_id
    input_ids[8 + L - 2 : 8 + L + C, 0] = cfg.tts_pad_token_id   # [8+L-2 : T]

    # ---- codec 通道 input_ids[:,1] ----
    input_ids[3:8, 1] = [
        cfg.codec_nothink_id,
        cfg.codec_think_bos_id,
        cfg.codec_think_eos_id,
        0,                       # speaker emb 槽（id=0 占位，训练步注入 speaker_emb）
        cfg.codec_pad_id,
    ]
    input_ids[8 : 8 + L - 3, 1] = cfg.codec_pad_id
    input_ids[8 + L - 3, 1] = cfg.codec_pad_id
    input_ids[8 + L - 2, 1] = cfg.codec_bos_id
    input_ids[8 + L - 1 : 8 + L - 1 + C, 1] = audio_codec_0          # [8+L-1 : 8+L-1+C]
    input_ids[8 + L - 1 + C, 1] = cfg.codec_eos_token_id            # pos T-1

    # ---- codec_0_labels（码本0 label，-100 mask） ----
    codec_0_labels[8 + L - 1 : 8 + L - 1 + C] = audio_codec_0
    codec_0_labels[8 + L - 1 + C] = cfg.codec_eos_token_id         # pos T-1

    # ---- codec_ids（全码本，喂 Code predictor） ----
    codec_ids[8 + L - 1 : 8 + L - 1 + C, :] = audio_codes

    # ---- masks ----
    text_embedding_mask[: 8 + L + C] = True                          # [:T]
    codec_embedding_mask[3 : 8 + L + C] = True                       # [3:T]
    codec_embedding_mask[6] = False                                   # speaker emb 槽
    codec_mask[8 + L - 1 : 8 + L - 1 + C] = True                      # [8+L-1 : 8+L+C-1) = C 个目标帧
    attention_mask[: 8 + L + C] = True                                # [:T]

    return {
        "input_ids": input_ids,
        "codec_ids": codec_ids,
        "codec_0_labels": codec_0_labels,
        "text_embedding_mask": text_embedding_mask,
        "codec_embedding_mask": codec_embedding_mask,
        "codec_mask": codec_mask,
        "attention_mask": attention_mask,
        "length": np.int64(T),
    }


def collate_qwen3_batch(
    samples: List[Dict[str, np.ndarray]],
    cfg: Qwen3TTSDataConfig,
    ref_mels: Optional[List[np.ndarray]] = None,
    max_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """把多个 build_qwen3_sample 输出 pad 成一个 batch（对齐官方 collate_fn 的 padding）。

    padding 规则（与官方一致）：
      input_ids/codec_ids -> 0；masks -> False；attention_mask -> 0；codec_0_labels -> -100。
    ref_mels（各样本 mel 帧数不同）按官方沿时间 cat（不 pad）—— speaker_encoder 内部池化。
    """
    b = len(samples)
    if max_length is None:
        max_length = int(max(int(s["length"]) for s in samples))
    t = max_length
    ncb = cfg.num_code_groups

    input_ids = np.zeros((b, t, 2), dtype=np.int64)
    codec_ids = np.zeros((b, t, ncb), dtype=np.int64)
    codec_0_labels = np.full((b, t), -100, dtype=np.int64)
    text_embedding_mask = np.zeros((b, t), dtype=bool)
    codec_embedding_mask = np.zeros((b, t), dtype=bool)
    codec_mask = np.zeros((b, t), dtype=bool)
    attention_mask = np.zeros((b, t), dtype=np.int64)

    for i, s in enumerate(samples):
        Ti = int(s["length"])
        input_ids[i, :Ti, :] = s["input_ids"]
        codec_ids[i, :Ti, :] = s["codec_ids"]
        codec_0_labels[i, :Ti] = s["codec_0_labels"]
        text_embedding_mask[i, :Ti] = s["text_embedding_mask"]
        codec_embedding_mask[i, :Ti] = s["codec_embedding_mask"]
        codec_mask[i, :Ti] = s["codec_mask"]
        attention_mask[i, :Ti] = s["attention_mask"]

    batch = {
        "input_ids": input_ids,
        "codec_ids": codec_ids,
        "codec_0_labels": codec_0_labels,
        "text_embedding_mask": text_embedding_mask[:, :, None],   # [b,t,1] 对齐官方
        "codec_embedding_mask": codec_embedding_mask[:, :, None],  # [b,t,1]
        "codec_mask": codec_mask,
        "attention_mask": attention_mask,
    }
    if ref_mels is not None:
        # 官方：torch.cat(ref_mels, dim=0) —— 各样本 mel 沿时间拼，speaker_encoder 池化后输出 [b, enc_dim]
        batch["ref_mels"] = np.concatenate(ref_mels, axis=0) if ref_mels else np.zeros((0,), dtype=np.float32)
    return batch
PYEOF
mkdir -p "$(dirname "tts/qwen3/forward.py")"
cat > "tts/qwen3/forward.py" <<'PYEOF'
"""Qwen3-TTS 自定义 forward 编排（W-2 附加 / W-4）。

移植自官方 `Qwen3-TTS/finetuning/sft_12hz.py` 的训练步 4 步手动流程（见 docs/05 第 5 节）。
Twinkle 标准 `model.forward_backward(inputs=batch, task='causal_lm')` 走"单次 model(**inputs)→loss"
路径，管不住 Qwen3-TTS 的：① speaker_encoder 前置 ② 手动组装 inputs_embeds ③ 2 段 forward。
故需自定义编排，本文件给出逻辑实现；Twinkle 集成方式见模块末"集成选项"。

W-4 变更：编排不再自己算总 loss，改为产出**中间态 dict**（talker_loss / sub_talker_logits /
sub_talker_labels / target_speaker_embedding），总 loss 由 Qwen3TTSLoss 从中间态算（职责清晰：
编排管前向，Loss 管 loss）。sub_talker 段取 forward_sub_talker_finetune 返回的 logits、
忽略其返回的 loss（继承的 ForCausalLMLoss 对 [N,15,V] 码本轴有 off-by-one，见 loss.py docstring）。

⚠️ 需要 Qwen3-TTS 真实模型权重才能跑（CPU 跑不动 forward），本文件为逻辑骨架 + W-5/W-6 在 NPU 验证。
"""

from typing import Dict, Optional
import torch


def qwen3_tts_forward_step(
    model,
    batch: Dict[str, torch.Tensor],
    target_speaker_embedding: Optional[torch.Tensor] = None,
):
    """Qwen3-TTS 训练一步前向（产出中间态 dict，供 Qwen3TTSLoss 算总 loss）。

    Args:
        model: `Qwen3TTSForConditionalGeneration`（底层，= `Qwen3TTSModel.model`），需含顶层
               `.speaker_encoder` + `.talker`（含 `.model.codec_embedding`、`.model.text_embedding`、
               `.text_projection`、`.codec_head`、`.code_predictor`、`.forward_sub_talker_finetune`）。
        batch: collate_qwen3_batch 输出转 torch 后的张量字典（input_ids[B,T,2]、codec_ids[B,T,16]、
               text_embedding_mask[B,T,1]、codec_embedding_mask[B,T,1]、codec_mask[B,T]、
               attention_mask[B,T]、codec_0_labels[B,T]、ref_mels[N,128]）。
        target_speaker_embedding: 首步算出的目标说话人 emb（官方写进 codec_embedding.weight[3000]；
               本项目走 clone 推理，可传 None，仅用于一致性记录）。

    Returns:
        dict:
          - 'talker_loss': 码本0 的 mean CE（model.talker.forward 的 outputs.loss，标准时间轴 shift 正确）。
          - 'sub_talker_logits': code_predictor 输出 [N, 15, V]（N=批次目标帧总数；15=码本1-15 头数）。
          - 'sub_talker_labels': [N, 15] = talker_codec_ids[:, 1:]（码本1-15 真值，已是 head 预测目标序列）。
          - 'target_speaker_embedding': 首步缓存的 speaker emb（供外部记录）。
        总 loss = talker_loss + 0.3 × code_predictor CE，由 Qwen3TTSLoss 从本 dict 算（不在编排里算）。
    """
    talker = model.talker
    # ---- ① speaker_encoder 前置（独立前向，detach 冻结）----
    speaker_embedding = model.speaker_encoder(batch["ref_mels"].to(model.dtype)).detach()
    if target_speaker_embedding is None:
        target_speaker_embedding = speaker_embedding

    # ---- ② 手动组装 inputs_embeds（不走 model(input_ids=...)）----
    input_text_ids = batch["input_ids"][..., 0]        # [B,T]
    input_codec_ids = batch["input_ids"][..., 1]      # [B,T]
    text_emb_mask = batch["text_embedding_mask"].to(model.dtype)
    codec_emb_mask = batch["codec_embedding_mask"].to(model.dtype)
    codec_ids = batch["codec_ids"]                    # [B,T,16]
    codec_mask = batch["codec_mask"]                  # [B,T]

    input_text_emb = talker.model.text_embedding(input_text_ids) * text_emb_mask
    input_codec_emb = talker.model.codec_embedding(input_codec_ids) * codec_emb_mask
    input_codec_emb[:, 6, :] = speaker_embedding      # 注入 pos6
    input_embeddings = input_text_emb + input_codec_emb

    # 叠加码本 1-15 的输入嵌入（code_predictor.get_input_embeddings() 返回 ModuleList）
    cp_input_embs = talker.code_predictor.get_input_embeddings()   # ModuleList[15]
    for i in range(1, talker.config.num_code_groups):
        e = cp_input_embs[i - 1](codec_ids[..., i]) * codec_mask.unsqueeze(-1).to(model.dtype)
        input_embeddings = input_embeddings + e

    # ---- ③ 第一段 forward：Talker 预测码本 0（inputs_embeds 与 labels 已 shift）----
    outputs = talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=batch["attention_mask"][:, :-1],
        labels=batch["codec_0_labels"][:, 1:],
        output_hidden_states=True,
    )
    talker_loss = outputs.loss                            # 码本0 mean shift+CE（HF loss_function，时间轴正确）

    # ---- ④ 第二段 forward：Code predictor 预测码本 1-15（向量化，一次喂所有目标帧）----
    # 注：forward_sub_talker_finetune 已向量化（输入 [N,16]+[N,H] → 内部 [N,16,H] 一次前向），
    #     非"逐帧循环"（docs/05 §3.2 / docs/06 §5.2 原述有误，W-4 修正）。
    hidden = outputs.hidden_states[0][-1]                 # [B, T-1, H]
    talker_hidden = hidden[codec_mask[:, :-1]]            # [N, H]，N=批次目标帧总数
    talker_codec_ids = codec_ids[codec_mask]             # [N, 16]
    sub_talker_logits, _sub_loss_ignored = talker.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden
    )
    # 忽略 _sub_loss_ignored（继承的 ForCausalLMLoss 对码本轴有 off-by-one，见 loss.py）；
    # Qwen3TTSLoss 用 sub_talker_logits + sub_talker_labels 自己算正确的 per-head CE。
    sub_talker_labels = talker_codec_ids[:, 1:]          # [N, 15] = 码本1-15 真值

    return {
        "talker_loss": talker_loss,
        "sub_talker_logits": sub_talker_logits,          # [N, 15, V]
        "sub_talker_labels": sub_talker_labels,           # [N, 15]
        "target_speaker_embedding": target_speaker_embedding,
    }


# --------------------------------------------------------------------
# Twinkle 集成选项（W-2 设计决策，待 W-5 NPU 验证后敲定）
# --------------------------------------------------------------------
# 选项 A（推荐）：自定义 TransformersModel 子类，override forward_backward/calculate_loss
#   - 编排：调 qwen3_tts_forward_step 拿中间态 dict，stash {'talker_loss','sub_talker_logits',
#     'sub_talker_labels'} 进 status.outputs（ModelOutput dict，自定义 key 运行时可塞）；
#     保留 clip_grad/梯度累积/metric 抽象 + 多 LoRA（adapter_name）透传。
#   - loss：set_loss(Qwen3TTSLoss)；calculate_loss 调 Qwen3TTSLoss(inputs, outputs) 从 stash 算总 loss。
#   - cookbook 仍调 model.forward_backward(inputs=batch)；编排全在子类内。
#   - 代价：要懂 twinkle.model.transformers.TransformersModel 的 forward/backward/calculate_loss
#     内部（transformers.py:851-869/757-805），override 时保留 clip_grad_and_step/梯度累积逻辑。
#
# 选项 B：自定义 task Patch（仿 patch/transformers_fused_ce.py 的 TransformersFusedCEPatch）
#   - 在 _resolve_task_context（transformers.py:85-103）注册新 task（如 'qwen3_tts'）；
#     patch 劫持 talker.forward 做 2 段 + speaker 前置 + stash 中间态。
#   - 优点：与 Twinkle task/Patch 范式一致；对 cookbook 透明（task='qwen3_tts'）。
#   - 代价：patch 劫持点要精准（需拿到 inputs_embeds 组装时机 + output_hidden_states）；多 LoRA 透传
#     需确认 patch 能访问 optimizer_group。
#
# 选项 C：绕过 forward_backward，cookbook 里直接调 qwen3_tts_forward_step + Qwen3TTSLoss 算 loss
#   + accelerator.backward（最接近官方 sft_12hz.py，但放弃 Twinkle 的 grad clip/lr step/metric 抽象，
#   最不"Twinkle"）。
#
# 倾向 A：编排复杂但可控，保留 Twinkle 训练循环抽象。W-5 去机房实测验证后定。
PYEOF
mkdir -p "$(dirname "tts/qwen3/input_processor.py")"
cat > "tts/qwen3/input_processor.py" <<'PYEOF'
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
PYEOF
mkdir -p "$(dirname "tts/qwen3/loss.py")"
cat > "tts/qwen3/loss.py" <<'PYEOF'
"""Qwen3-TTS Loss（cookbook，W-4 实现）。

Qwen3 是 2 段式：loss = talker CE(码本0) + 0.3 × code_predictor CE(码本1-15)。

- talker 段（码本0，[B,T,V] 标准时间轴 next-token）：编排从 model.talker.forward 拿 outputs.loss
  （HF 继承的 ForCausalLMLoss 对时间轴的 shift 是正确的），本类透传。
- code_predictor 段（码本1-15，[N,15,V] 沿码本轴 depth-transformer next-token）：本类显式算
  per-head CE。

⚠️ W-4 实测确认 off-by-one（见 docs/05 §5/§8 + tts/qwen3/tests/test_loss.py）：
  官方 forward_finetune（modeling_qwen3_tts.py:1242）对 [B,15,V] logits 调继承的
  self.loss_function（=ForCausalLMLoss），它在最后一轴做标准 LM shift：
    shift_labels = pad(labels,(0,1),-100)[...,1:]
  导致 head j（用 pos(j+1) 的 hidden 预测码本 j+1）被错配到码本 j+2 的 label，
  且 cb15 的预测被末位 -100 pad 吞掉（不算 loss），且 cb1 监督被首切丢失
  （labels[...,1:] 丢 labels[:,0]=cb1）——**双重偏移**：forward_sub_talker_finetune
  已把 labels 预处理成 codec_ids[:,1:]（预对齐，labels[:,j]=cb[j+1]，和 logits[:,j]
  天然对齐），ForCausalLMLoss 又 shift 一次。**这是 Qwen3-TTS 官方 bug**（会话9 Task agent
  读 modeling_qwen3_tts.py:1612-1633 源码确认），待本项目做完有 WER/SIM 证据时提 bugfix PR。
  同位 per-head CE（logits[:,j] ↔ labels[:,j]=码本 j+1，无 shift）才是 depth-transformer
  next-token 的正确语义——depth-transformer 的"next-token"已体现在"pos(j+1) 输入码本 j emb，
  head 读 pos(j+1) hidden 预测码本 j+1"里，labels 已是 head 预测目标序列，不需再 shift。

shift_labels 开关（W-5 NPU 实测定默认）：
  - False（默认，修正）：同位 per-head CE，不 shift。
  - True（复现官方）：复刻 ForCausalLMLoss 的 shift（含 off-by-one），供 W-5 对照。

集成（选项A 自定义 TransformersModel 包装器，W-5 落实）：编排 qwen3_tts_forward_step 产出
中间态 dict，包装器把 {'talker_loss','sub_talker_logits','sub_talker_labels'} stash 进
status.outputs；本类从 outputs 读这些算总 loss（total 不在编排里算，职责清晰）。
"""

import torch
import torch.nn.functional as F

from twinkle.data_format import LossOutput

from tts.common import MultiCodebookLoss


def _shift_ce_like_official(logits, labels, ignore_index=-100):
    """复刻 transformers ForCausalLMLoss 的 shift 语义（含 off-by-one，仅供 shift_labels=True 对照）。

    logits [N, C, V]，labels [N, C]。
    """
    from transformers.loss.loss_utils import ForCausalLMLoss

    return ForCausalLMLoss(
        logits=logits,
        labels=labels,
        vocab_size=logits.shape[-1],
        ignore_index=ignore_index,
    )


class Qwen3TTSLoss(MultiCodebookLoss):
    """Qwen3-TTS 2 段式 loss。

    Args:
        sub_loss_weight: code_predictor 段权重（官方 0.3）。
        shift_labels: code_predictor 段是否做标准 LM shift。
            False（默认）= 修正 off-by-one，同位 per-head CE（无 shift）；
            True = 复现官方 ForCausalLMLoss 含 shift（W-5 实测对照用）。
        ignore_index: 忽略的 label 值（非音频帧 / pad）。
    """

    def __init__(
        self,
        sub_loss_weight: float = 0.3,
        shift_labels: bool = False,
        ignore_index: int = -100,
        ar_codebook: int = 0,
    ):
        super().__init__(ar_codebook=ar_codebook)
        self.sub_loss_weight = sub_loss_weight
        self.shift_labels = shift_labels
        self.ignore_index = ignore_index

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        # talker 段（码本0）：透传编排算好的 mean CE（标准时间轴 shift 正确）
        talker_loss = outputs["talker_loss"]
        # code_predictor 段（码本1-15）
        sub_logits = outputs["sub_talker_logits"]   # [N, 15, V]
        sub_labels = outputs["sub_talker_labels"]   # [N, 15]

        if self.shift_labels:
            sub_loss = _shift_ce_like_official(sub_logits, sub_labels, self.ignore_index)
        else:
            V = sub_logits.shape[-1]
            # 同位 per-head CE（无 shift）：logits[:,j] 预测码本 j+1 ↔ labels[:,j]=码本 j+1
            sub_loss = F.cross_entropy(
                sub_logits.reshape(-1, V),
                sub_labels.reshape(-1),
                ignore_index=self.ignore_index,
                reduction="mean",
            )

        total = talker_loss + self.sub_loss_weight * sub_loss
        # 两段均 per-token mean（镜像官方 sft_12hz.py：outputs.loss 与 sub_loss 都是 mean）。
        # num_tokens=0 走 Twinkle PER-TOKEN-MEAN 归一（grad_accum 路径），
        # 与 accelerate.backward(loss)+gradient_accumulation_steps 一致。
        return LossOutput(loss=total, num_tokens=0)
PYEOF
mkdir -p "$(dirname "tts/qwen3/model.py")"
cat > "tts/qwen3/model.py" <<'PYEOF'
"""Qwen3-TTS 自定义 TransformersModel 子类（选项A，W-5 预写骨架，待 NPU 实测定稿）。

Twinkle 标准 `TransformersModel.forward` 走"单次 `self.model(**inputs)` → 算 logps → 写 status"
路径（`transformers.py:514-630`），管不住 Qwen3-TTS 的 speaker_encoder 前置 + 手动组装
inputs_embeds + 2 段 forward（talker 预测码本0、code_predictor 预测码本1-15）。故子类
override `forward`，把中间那步 `self.model(**inputs)` 换成 `qwen3_tts_forward_step`，
后者产出**中间态 dict**（talker_loss / sub_talker_logits / sub_talker_labels）写进
`status.outputs`；`calculate_loss` / `backward` **复用基类**——因为 `Qwen3TTSLoss.__call__`
（`loss.py:75`）直接从 `outputs` 读这三个键算总 loss，而基类 `calculate_loss`
（`transformers.py:757-805`）正好把 `status.outputs` 传给 `loss_instance`。

跳过的基类 forward 分支（Qwen3-TTS 不用，W-5 现场确认跳过不破坏）：
- `labels = inputs.pop('labels')` + `selective_log_softmax` 的 logps 段：`Qwen3TTSLoss`
  设 `require_logps=False`（`MultiCodebookLoss` 类属性），自算 CE，不读 `outputs['logps']`。
- sampling replay / router replay / `_resolve_task_context`（task patch）：TTS 训练不走
  causal_lm 的 task patch（选项A 在子类 override 前向，非选项B 的 patch 劫持），LoRA 透传
  由 `PeftModel` 包装（`add_adapter_to_model` 挂在 talker/code_predictor 各层）保证，无需 task context。
- `disable_lora` / `return_logits` 控制：TTS 训练不 disable LoRA；中间态不含 'logits' 键。

⚠️ 待实测点（对照 twinkle/src/twinkle/model/transformers/transformers.py 基类 forward:515-630 /
forward_backward:852-869 / calculate_loss:758-805 读完精化；代码逻辑已验正确，以下是 NPU 实测项）：
1. `@remote_function` 装饰器在子类 override 上是否生效（remote 机制对 override 方法的行为）。
2. `forward_only`（eval 路径，transformers.py:633）**本会话不 override**：基类 forward_only 走单次
   `self.model(**inputs)`，对 Qwen3-TTS 2 段编排会崩；训练脚本 eval 暂跳过，留 WER/SIM 阶段再补
   override（同样替换为 qwen3_tts_forward_step）。
3. ✅ 已确认：`status.loss_value = 0`（行 92）与基类 `outputs.get('aux_loss', 0)`（transformers.py:624）
   在无 aux_loss 场景等价——calculate_loss(802) 从 0 累加 Qwen3TTSLoss 的 loss，backward(818) 用之。
4. `self.model` 经 `_lazy_wrap_model()`（transformers.py:394）后被 strategy(Accelerate/FSDP)+PeftModel
   多层包装，`model.talker`/`model.speaker_encoder`/`model.code_predictor` 能否经各层 getattr 透传到
   底层 Qwen3TTSForConditionalGeneration（qwen3_tts_forward_step 要的）。
5. 【新】`optimizer_config.processor(inputs)`（行 74）对「已 collate 的 2D batch dict」的行为：训练脚本
   （train_qwen3_tts_lora.py）走 smoke-test 简化路径，传 collate_qwen3_batch 产出的已 pad 2D batch
   （有 input_ids，_not_encoded=False 跳过 template.batch_encode）。default InputProcessor 对已 collate
   2D batch 是否正确 to_transformers_dict 未验——若报错，set Qwen3TTSInputProcessor 或本类对已 collate
   batch 跳过 processor（直接进 qwen3_tts_forward_step）。
6. 【新】跳过基类 `postprocess_tensor_sp`/`unpack_packed_sequences`（transformers.py:619-620）：单卡
   无 SP（sequence parallel）无 packed sequences，应可跳过；实测确认对 2D batch 无必需处理。

架构见 `docs/05` 第 5 节；集成选项见 `tts/qwen3/forward.py` 末尾；Loss 读中间态见 `tts/qwen3/loss.py`。
"""

from copy import copy

from twinkle import remote_function
from twinkle.data_format import InputFeature, Trajectory
from twinkle.infra import collect_tensor_dict
from twinkle.model.transformers import TransformersModel

from .forward import qwen3_tts_forward_step


class Qwen3TTSModel(TransformersModel):
    """选项A：override `forward` 走 Qwen3-TTS 2 段编排 + stash 中间态。

    `calculate_loss` / `backward` / `clip_grad_norm` 等复用基类（保留梯度累积/clip/metric 抽象
    + 多 LoRA 透传）。cookbook 仍调 `model.forward_backward(inputs=batch)`，编排全在本类内。
    """

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict)
    def forward(self, *, inputs, **kwargs):
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        # task 不用 causal_lm，但 pop 掉避免传给下游 qwen3_tts_forward_step
        kwargs.pop('task', None)
        kwargs.pop('temperature', 1.0)
        kwargs.pop('sampling_masks', None)
        kwargs.pop('return_logits', False)
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        if not inputs:
            raise ValueError('inputs empty, check your DataLoader outputs')
        self.model.train()

        # 1. Trajectory / List[Trajectory] → InputFeature（若未编码）
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (
            isinstance(inputs, list) and self._not_encoded(inputs[0])
        ):
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)

        # 2. InputFeature → batch dict（复用基类 InputProcessor，2D stack + ref_mels cat）
        processor = optimizer_config.processor
        inputs = processor(
            inputs,
            sp_strategy=self.sp_strategy,
            model=self.model,
            hf_config=self.hf_config,
            enable_sp=getattr(self, '_enable_sp', False),
        )

        # 3. Qwen3-TTS 2 段前向编排（替换基类的 `self.model(**inputs)`）
        #    产出中间态 dict：talker_loss / sub_talker_logits / sub_talker_labels /
        #    target_speaker_embedding。
        outputs = qwen3_tts_forward_step(self.model, inputs)

        # 4. stash 中间态进 train_status.outputs，供 Qwen3TTSLoss 从 outputs 读（calculate_loss 复用基类）
        status = optimizer_config.train_status
        status.inputs = inputs
        status.outputs = outputs
        status.forward_kwargs = kwargs
        status.loss_value = 0
        return copy(outputs)
PYEOF
mkdir -p "$(dirname "tts/qwen3/processor.py")"
cat > "tts/qwen3/processor.py" <<'PYEOF'
"""Qwen3-TTS Processor（cookbook）。

Qwen3 官方 manifest 字段名（audio/text/ref_audio/language）正好匹配通用基类默认值，
故 `Qwen3TTSProcessor` 直接继承 `MultiCodebookPreprocessor` 即可——仅作语义标记 +
将来若 Qwen3 字段名有偏差时在此覆盖。真正模型特定的重活在 `Qwen3TTSTemplate`
（构建双通道 InputFeature，见 template.py）。
"""

from tts.common import MultiCodebookPreprocessor


class Qwen3TTSProcessor(MultiCodebookPreprocessor):
    """manifest 行 → Trajectory（Qwen3-TTS 字段名与通用基类一致）。"""

    # Qwen3 官方 manifest（Qwen3-TTS/finetuning/prepare_data.py 输入）字段名：
    # {"audio":..., "text":..., "ref_audio":..., "audio_codes":[[...]], "language":...}
    # 与 MultiCodebookPreprocessor 默认完全一致，无需覆盖。
    pass
PYEOF
mkdir -p "$(dirname "tts/qwen3/template.py")"
cat > "tts/qwen3/template.py" <<'PYEOF'
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
PYEOF
mkdir -p "$(dirname "tts/qwen3/tests/test_data.py")"
cat > "tts/qwen3/tests/test_data.py" <<'PYEOF'
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
PYEOF
mkdir -p "$(dirname "tts/qwen3/tests/test_input_processor.py")"
cat > "tts/qwen3/tests/test_input_processor.py" <<'PYEOF'
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
PYEOF
mkdir -p "$(dirname "tts/qwen3/tests/test_loss.py")"
cat > "tts/qwen3/tests/test_loss.py" <<'PYEOF'
"""Qwen3TTSLoss 单测（W-4，CPU，不需真实模型）。

验证：
1. 默认 shift_labels=False（修正）：同位 per-head CE，"同位预测正确"时 loss≈0。
2. shift_labels=True（复现官方）：sub_loss == ForCausalLMLoss(logits, labels, V)（含 off-by-one）。
3. off-by-one 对比：同一组"同位正确"logits，修正模式 loss≈0、复现模式 loss 大。
4. 加权：total = talker_loss + 0.3 × sub_loss。
5. ignore_index：-100 位置不算。
6. cb15：修正模式计入（loss 受 cb15 影响）；复现模式被末位 -100 吞（不受 cb15 影响）。
7. num_tokens=0。
8. 多帧 N=3。

运行：PYTHONPATH=twinkle/src python tts/qwen3/tests/test_loss.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))                       # 项目根（import tts）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "twinkle", "src"))      # twinkle

from tts.qwen3.loss import Qwen3TTSLoss, _shift_ce_like_official


def _correct_logits(labels, V, big=20.0):
    """logits[n,j, labels[n,j]] = +big，其余 -big → 同位预测自信正确。"""
    N, C = labels.shape
    logits = torch.full((N, C, V), -big)
    for n in range(N):
        for j in range(C):
            logits[n, j, labels[n, j].item()] = big
    return logits


def _outputs(talker_loss, sub_logits, sub_labels):
    return {
        "talker_loss": torch.tensor(float(talker_loss)),
        "sub_talker_logits": sub_logits,
        "sub_talker_labels": sub_labels,
    }


def test_default_same_position_correct():
    labels = torch.arange(1, 16).unsqueeze(0)   # [1,15] = 码本1..15
    V = 20
    logits = _correct_logits(labels, V)
    out = Qwen3TTSLoss()(inputs={}, outputs=_outputs(0.0, logits, labels))
    assert out["loss"].item() < 1e-4, out["loss"].item()
    assert out["num_tokens"] == 0
    print("  [OK] 默认修正模式：同位预测正确 → loss≈0, num_tokens=0")


def test_shift_mode_equals_forcausallmloss():
    torch.manual_seed(0)
    labels = torch.arange(1, 16).unsqueeze(0)
    V = 20
    logits = torch.randn(1, 15, V)
    out = Qwen3TTSLoss(shift_labels=True)(inputs={}, outputs=_outputs(0.0, logits, labels))
    expected = _shift_ce_like_official(logits, labels)
    assert torch.allclose(out["loss"], 0.3 * expected, atol=1e-5), (out["loss"], 0.3 * expected)
    print("  [OK] shift_labels=True：sub_loss == ForCausalLMLoss（含 off-by-one）")


def test_off_by_one_contrast():
    labels = torch.arange(1, 16).unsqueeze(0)
    V = 20
    logits = _correct_logits(labels, V)   # 同位预测正确
    loss_fix = Qwen3TTSLoss(shift_labels=False)(inputs={}, outputs=_outputs(0.0, logits, labels))["loss"]
    loss_off = Qwen3TTSLoss(shift_labels=True)(inputs={}, outputs=_outputs(0.0, logits, labels))["loss"]
    assert loss_fix.item() < 1e-4, loss_fix.item()
    assert loss_off.item() > 5.0, loss_off.item()    # 错位 → loss 大
    print(f"  [OK] off-by-one 对比：修正 loss={loss_fix.item():.4f} ≈0；复现 loss={loss_off.item():.4f} 大")


def test_weighting():
    torch.manual_seed(1)
    labels = torch.arange(1, 16).unsqueeze(0)
    V = 20
    logits = torch.randn(1, 15, V)
    talker = torch.tensor(1.5)
    out = Qwen3TTSLoss(sub_loss_weight=0.3)(inputs={}, outputs=_outputs(talker, logits, labels))
    sub = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100, reduction="mean")
    expected = talker + 0.3 * sub
    assert torch.allclose(out["loss"], expected, atol=1e-5), (out["loss"], expected)
    print(f"  [OK] 加权：total = talker({talker.item()}) + 0.3×sub({sub.item():.4f}) = {expected.item():.4f}")


def test_ignore_index():
    labels = torch.tensor([[1, 2, -100, 4, 5, -100, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    V = 20
    logits = _correct_logits(labels.clamp(min=0), V)   # -100 位置 clamp 到 0（反正被 ignore）
    out = Qwen3TTSLoss()(inputs={}, outputs=_outputs(0.0, logits, labels))
    # 有效 13 个位置都预测正确 → loss≈0
    assert out["loss"].item() < 1e-4, out["loss"].item()
    print("  [OK] ignore_index：-100 位置不计入，有效位置全对 → loss≈0")


def test_cb15_masked_only_in_shift_mode():
    # 构造：码本1-14 同位预测正确，唯独 cb15（labels[:,14]=15）预测错
    labels = torch.arange(1, 16).unsqueeze(0)
    V = 20
    logits = _correct_logits(labels, V)
    # 把 cb15 位置改成预测错（指向类别 0 而非 15）
    logits[0, 14, 15] = -20.0
    logits[0, 14, 0] = 20.0
    loss_fix = Qwen3TTSLoss(shift_labels=False)(inputs={}, outputs=_outputs(0.0, logits, labels))["loss"]
    loss_off = Qwen3TTSLoss(shift_labels=True)(inputs={}, outputs=_outputs(0.0, logits, labels))["loss"]
    # 修正模式：cb15 计入 → 有错 → loss 大（0.3 加权后 ≈ 0.3×40/15 ≈ 0.8）
    assert loss_fix.item() > 0.5, loss_fix.item()
    # 复现模式：cb15（logits[:,14]）被 shift 后配到末位 -100 → 不算 cb15 的错；
    #          但 cb1-14 也因 shift 错位 → loss 大（来自错位，不是 cb15）
    # 关键：复现模式里 cb15 的"错"不贡献 loss（被吞）。
    # 验证方式：把 cb15 改回正确，复现模式 loss 不应变（因 cb15 本就被吞）。
    logits_ok_cb15 = _correct_logits(labels, V)   # cb15 也对
    loss_off_ok = Qwen3TTSLoss(shift_labels=True)(inputs={}, outputs=_outputs(0.0, logits_ok_cb15, labels))["loss"]
    assert torch.allclose(loss_off, loss_off_ok, atol=1e-5), (loss_off.item(), loss_off_ok.item())
    print(f"  [OK] cb15：修正模式计入(loss={loss_fix.item():.4f}>0)；复现模式吞掉 cb15(改cb15对/错 loss 不变={loss_off.item():.4f})")


def test_multi_frame_N3():
    torch.manual_seed(2)
    labels = torch.stack([
        torch.arange(1, 16),
        torch.arange(1, 16) + 1,
        torch.arange(1, 16) + 2,
    ])   # [3, 15]，每帧码本不同
    labels = labels.clamp(max=19)
    V = 20
    logits = _correct_logits(labels, V)
    out = Qwen3TTSLoss()(inputs={}, outputs=_outputs(0.0, logits, labels))
    assert out["loss"].item() < 1e-4, out["loss"].item()
    # 对照手算 mean
    sub = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100, reduction="mean")
    assert torch.allclose(out["loss"], 0.3 * sub, atol=1e-5)
    print(f"  [OK] 多帧 N=3：同位预测正确 → loss≈0；与手算 mean 一致")


def main():
    tests = [
        test_default_same_position_correct,
        test_shift_mode_equals_forcausallmloss,
        test_off_by_one_contrast,
        test_weighting,
        test_ignore_index,
        test_cb15_masked_only_in_shift_mode,
        test_multi_frame_N3,
    ]
    print(f"Qwen3TTSLoss 单测（{len(tests)} 项）")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\n全部通过（{len(tests)} 项）")


if __name__ == "__main__":
    main()
PYEOF
mkdir -p "$(dirname "tts/qwen3/tests/test_template.py")"
cat > "tts/qwen3/tests/test_template.py" <<'PYEOF'
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
PYEOF
mkdir -p "$(dirname "scripts/analyze_dataset_quality.py")"
cat > "scripts/analyze_dataset_quality.py" <<'PYEOF'
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
PYEOF
mkdir -p "$(dirname "scripts/prepare_aishell3.py")"
cat > "scripts/prepare_aishell3.py" <<'PYEOF'
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
PYEOF
mkdir -p "$(dirname "scripts/probe_api.py")"
cat > "scripts/probe_api.py" <<'PYEOF'
"""探明 qwen_tts / twinkle 的关键 API（点亮 prepare_aishell3.py / train_qwen3_tts_lora.py 的 TODO）。

进容器**最该先跑**这个——本机无法验（qwen_tts/twinkle 只在 NPU 容器）。跑完把输出
贴给上游会话，据结果填进 prepare/train 的 TODO。对照 `docs/05` §2.1（encode）/§3.3（mel）/§3.1（模型）。

点亮的 TODO：
  - prepare: T1（Qwen3TTSTokenizer 导入+encode 签名）/ T2（extract_mels 路径）
  - train:   L1（Qwen3TTSForConditionalGeneration 导入）/ L2（config 加载）/ L3（文本 tokenizer）/ L4（scheduler 名）

运行（容器内）：
  python scripts/probe_api.py                                    # 默认探全（不含 encode 实跑）
  python scripts/probe_api.py --tokenizer_path /path/to/Tokenizer-12Hz   # 加测 encode 签名
  python scripts/probe_api.py --model_path /home/h00875519/Qwen3-TTS-12Hz-1.7B-Base
"""

import argparse
import importlib
import inspect
import pkgutil

MODEL_DEFAULT = "/home/h00875519/Qwen3-TTS-12Hz-1.7B-Base"


def probe_import_candidates():
    """T1 + L1：候选导入路径逐个试，打印可 import 的类名。"""
    print("\n===== T1/L1: 候选模块 import 检查 =====")
    candidates = [
        ("Qwen3TTSTokenizer", "qwen_tts.core.tokenize.speech_tokenizer"),
        ("Qwen3TTSModel 高层", "qwen_tts.inference.qwen3_tts_model"),
        ("Qwen3TTSForConditionalGeneration 底层", "qwen_tts.core.models.modeling_qwen3_tts"),
        ("Qwen3TTSConfig", "qwen_tts.core.models.configuration_qwen3_tts"),
    ]
    for label, mod in candidates:
        try:
            m = importlib.import_module(mod)
            names = [n for n in dir(m) if "Qwen3" in n or "Tokenizer" in n]
            print(f"  [OK]   {label:40s} {mod}  -> {names}")
        except Exception as e:
            print(f"  [FAIL] {label:40s} {mod}  -> {e}")


def probe_extract_mels():
    """T2：全局搜 extract_mel 函数（ECAPA-TDNN 配套 mel 提取，格式一致性关键）。"""
    print("\n===== T2: extract_mels 全局搜索 =====")
    try:
        import qwen_tts
    except Exception as e:
        print(f"  [FAIL] import qwen_tts: {e}")
        return
    found = False
    for mod_info in pkgutil.walk_packages(qwen_tts.__path__, "qwen_tts."):
        try:
            m = importlib.import_module(mod_info.name)
        except Exception:
            continue
        for n in dir(m):
            obj = getattr(m, n)
            if callable(obj) and ("extract_mel" in n.lower() or (
                    "mel" in n.lower() and "extract" in n.lower())):
                try:
                    sig = inspect.signature(obj)
                except (ValueError, TypeError):
                    sig = "(?)"
                print(f"  [MEL] {mod_info.name}.{n}{sig}")
                found = True
    if not found:
        print("  未找到 extract_mel——fallback 用 torchaudio kaldi.fbank（prepare 脚本已给）")


def probe_config_load(model_path):
    """L2：AutoConfig 是否认 qwen3_tts；不认则用 Qwen3TTSConfig。"""
    print("\n===== L2: config 加载方式 =====")
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        has_tc = hasattr(cfg, "talker_config")
        print(f"  [OK] AutoConfig 能加载: type={type(cfg).__name__}, has talker_config={has_tc}")
        if has_tc:
            ncb = getattr(cfg, "num_code_groups", "无")
            print(f"       num_code_groups={ncb}（data.py 默认 16，{'一致' if ncb == 16 else '需同步'}）")
            print("  -> train 脚本可直接 TransformersModel(model_id=..., config=AutoConfig 结果)")
    except Exception as e:
        print(f"  [FAIL] AutoConfig 不认 qwen3_tts: {e}")
        print("  -> 用 Qwen3TTSConfig.from_pretrained(model_path) 显式传 config= 参数")


def probe_text_tokenizer():
    """L3：搜 qwen_tts 的文本 tokenizer（Qwen3 文本侧，非语音 codec）。"""
    print("\n===== L3: 文本 tokenizer 获取方式 =====")
    try:
        import qwen_tts
    except Exception:
        print("  [FAIL] qwen_tts 未装")
        return
    found = []
    for mod_info in pkgutil.walk_packages(qwen_tts.__path__, "qwen_tts."):
        try:
            m = importlib.import_module(mod_info.name)
        except Exception:
            continue
        for n in dir(m):
            if "TextTokenizer" in n or ("Tokenizer" in n and "Speech" not in n and "Codec" not in n):
                obj = getattr(m, n)
                if isinstance(obj, type):
                    found.append(f"{mod_info.name}.{n}")
    if found:
        print("  候选文本 tokenizer 类：")
        for f in found:
            print(f"    {f}")
        print("  -> train 脚本：模型加载后试 model.tokenizer；若无则用上面类 from_pretrained")
    else:
        print("  未找到独立文本 tokenizer——可能 model.tokenizer 直接可用（加载后试）")


def probe_scheduler():
    """L4：twinkle 支持的 scheduler 名（train 脚本 set_lr_scheduler 的字符串）。"""
    print("\n===== L4: twinkle scheduler 名 =====")
    try:
        import twinkle
    except Exception as e:
        print(f"  [FAIL] import twinkle: {e}")
        return
    # 搜 twinkle 下的 scheduler 相关模块/类
    sched_names = set()
    for attr in dir(twinkle):
        if "sched" in attr.lower():
            sched_names.add(f"twinkle.{attr}")
    # 试常见的 scheduler 模块
    for mod_name in ("twinkle.lr_scheduler", "twinkle.scheduler", "twinkle.optim.scheduler",
                      "twinkle.optimizer.scheduler"):
        try:
            m = importlib.import_module(mod_name)
            for n in dir(m):
                obj = getattr(m, n)
                if isinstance(obj, type) and ("Sched" in n or "LR" in n):
                    sched_names.add(f"{mod_name}.{n}")
        except Exception:
            pass
    if sched_names:
        print("  候选 scheduler：")
        for s in sorted(sched_names):
            print(f"    {s}")
        print("  -> train 脚本 set_lr_scheduler(scheduler_cls='上面类名' 或 'CosineWarmupScheduler')")
    else:
        print("  未直接找到——查 twinkle 源码 src/twinkle/ 下 scheduler 模块")


def probe_encode_signature(tokenizer_path):
    """T1 实跑：Qwen3TTSTokenizer.encode 签名（需下 speech_tokenizer 权重）。"""
    print("\n===== T1 实跑: Qwen3TTSTokenizer.encode 签名 =====")
    if not tokenizer_path:
        print("  跳过（未传 --tokenizer_path）——传 {model_path}/speech_tokenizer 子目录再跑")
        return
    # probe 已确认导入路径（modeling_qwen3_tts.Qwen3TTSTokenizer）
    for mod_path, cls_name in [
        ("qwen_tts.core.models.modeling_qwen3_tts", "Qwen3TTSTokenizer"),
    ]:
        try:
            m = importlib.import_module(mod_path)
            cls = getattr(m, cls_name)
            print(f"  尝试 {mod_path}.{cls_name}")
            try:
                fp_sig = inspect.signature(cls.from_pretrained)
                print(f"    from_pretrained{fp_sig}")
            except (ValueError, TypeError):
                print("    from_pretrained 签名不可读")
            tok = cls.from_pretrained(tokenizer_path)
            print(f"    加载成功: {type(tok).__name__}")
            try:
                enc_sig = inspect.signature(tok.encode)
                print(f"    encode{enc_sig}")
            except (ValueError, TypeError):
                print("    encode 签名不可读——试 help(tok.encode)")
            # 打印返回的 .audio_codes 属性提示（docs/05 §2.1）
            print("    返回应有 .audio_codes → [T,16]（docs/05 §2.1）")
            return
        except Exception as e:
            print(f"    [FAIL] {e}")
    print("  所有候选路径失败——看上面 T1/L1 的 [OK] 模块，找 Qwen3TTSTokenizer 真实路径")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=MODEL_DEFAULT,
                    help=f"Qwen3-TTS 权重路径（默认 {MODEL_DEFAULT}）")
    ap.add_argument("--tokenizer_path", default=None,
                    help="speech_tokenizer 权重路径（传了才实跑 encode 签名）")
    args = ap.parse_args()

    print(f"model_path = {args.model_path}")
    print(f"tokenizer_path = {args.tokenizer_path or '(未传，跳过 encode 实跑)'}")

    probe_import_candidates()
    probe_extract_mels()
    probe_config_load(args.model_path)
    probe_text_tokenizer()
    probe_scheduler()
    probe_encode_signature(args.tokenizer_path)

    print("\n===== 完成 =====")
    print("把上面输出贴给上游会话，据结果填 prepare_aishell3.py(T1/T2) + train_qwen3_tts_lora.py(L1/L2/L3/L4) 的 TODO")


if __name__ == "__main__":
    main()
PYEOF
mkdir -p "$(dirname "scripts/train_qwen3_tts_lora.py")"
cat > "scripts/train_qwen3_tts_lora.py" <<'PYEOF'
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
PYEOF

echo "  铺好的文件："
find "$INSTALL_DIR/tts" "$INSTALL_DIR/scripts" -name '*.py' 2>/dev/null | sort | sed 's/^/    /'

# ---- 段2 回归单测 ----
echo "[2/3] 跑 tts/ 回归单测..."
cd "$INSTALL_DIR"
for t in test_data test_template test_input_processor test_loss; do
    echo "  --- $t ---"
    python "tts/qwen3/tests/$t.py"
done
echo "  单测回归完成（test_template 的 real-tokenizer 集成测在权重未下时 SKIP，属正常）"

# ---- 段3 选项A 子类 import 验证 ----
echo "[3/3] 选项A 子类 + 组件 import 验证..."
python - <<'PYIMP'
from tts.qwen3.model import Qwen3TTSModel
from tts.qwen3.loss import Qwen3TTSLoss
from tts.qwen3.forward import qwen3_tts_forward_step
from tts.qwen3.template import Qwen3TTSTemplate
from tts.qwen3.input_processor import Qwen3TTSInputProcessor
from tts.qwen3.processor import Qwen3TTSProcessor
from tts.qwen3.data import Qwen3TTSDataConfig, build_qwen3_sample, collate_qwen3_batch
from tts.common import MultiCodebookPreprocessor, MultiCodebookLoss
print("  OK   Qwen3TTSModel / Qwen3TTSLoss / qwen3_tts_forward_step / Template / InputProcessor / Processor / data / common 全部 import 通")
PYIMP

echo
echo "================================================"
echo " 就绪。下一步 LoRA SFT 实测（见 docs/10-容器实测runbook.md）："
echo "   1) 探明 qwen_tts API（Qwen3TTSTokenizer.encode / Qwen3TTSForConditionalGeneration / extract_mels）"
echo "   2) 下 AISHELL-3 + 填 scripts/prepare_aishell3.py TODO → train.jsonl/dev.jsonl"
echo "   3) 填 scripts/train_qwen3_tts_lora.py TODO → --max_steps 2 smoke-test 看 loss 不崩"
echo "   4) 跑 w5_verify_tts.py 回归确认前向 OK"
echo "================================================"
