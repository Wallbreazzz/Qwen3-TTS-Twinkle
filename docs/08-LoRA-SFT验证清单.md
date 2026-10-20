# LoRA SFT 训练：本会话产出与容器实测清单（会话10）

> 本会话（会话10）在**本地写好** LoRA SFT 训练的代码骨架，未进容器。下个会话进卡7
> 容器点亮 TODO + 实测。本文件是进容器的执行指南。
> 术语见 `docs/01`；硬规则见 `AGENTS.md`；进度权威见 `新会话必读.md` 第8节。

## 1. 本会话产出（本地写好，未实测）

| 文件 | 作用 | 状态 |
|---|---|---|
| `scripts/prepare_aishell3.py` | AISHELL-3 单说话人 → Qwen3-TTS manifest 预编码（升采样/mel/codec→train.jsonl+dev.jsonl） | 骨架，T1/T2/T3 已填（会话12 probe），T4 gap 保留 |
| `scripts/train_qwen3_tts_lora.py` | Twinkle LoRA SFT 训练脚本（仿 fsdp2，单卡 NPU，含 LoRA alpha=2r 配置） | 骨架，L1-L4 已填（会话12 probe），L5/L6 待 smoke-test |
| `tts/qwen3/model.py` | 选项A `Qwen3TTSModel`（override forward 走 2 段编排），TODO 段精化 | 逻辑已验，M3 已确认，M1/M2/M4/M5/M6 待 smoke-test |

复用的已验证组件（不改）：`qwen3_tts_forward_step`（forward.py）、`Qwen3TTSLoss`（loss.py，
shift=False 修正 off-by-one）、`build_qwen3_sample`/`collate_qwen3_batch`（data.py）、
`w5_verify_tts.py` 的模型加载逻辑。

## 2. 为什么 W-5 之后还有开发（边界澄清）

W-5 用 `w5_verify_tts.py` **直接调** `qwen3_tts_forward_step`+`Qwen3TTSLoss`（绕过 Twinkle 训练
循环），只验**组件单元逻辑**（前向不崩/形状对/数值合理/off-by-one），ref_mels 用随机占位。
本会话做的是把零件装成机器——model.py override 进 Twinkle `forward_backward` + 训练脚本 +
LoRA 挂到 TTS 多模块 + 真实数据流水线。model.py docstring 自标"W-5 预写骨架，待定稿"。

**本会话未做**（下个会话）：进容器实测、跑真实训练看 loss 降、WER/SIM 评估、部署。

## 3. 容器实测 TODO 清单（按文件，进容器逐条点亮）

### 3.1 `scripts/prepare_aishell3.py`（数据流水线）
| TODO | 做什么 | 对照源 |
|---|---|---|
| T1 | ✅ 已确认（probe 实跑）：导入 `qwen_tts.core.models.modeling_qwen3_tts.Qwen3TTSTokenizer`；权重在 `{model}/speech_tokenizer/` 子目录；`encode(audios, sr=None, return_dict=True)` 接受 wav ndarray + sr。已填 `encode_audio_codes` | `docs/05` §2.1 + probe |
| T2 | ✅ 已确认（probe）：qwen_tts 无官方 `extract_mels`，fallback 用 torchaudio kaldi.fbank（ECAPA-TDNN 格式一致性待 smoke-test 验） | `docs/05` §3.3 |
| T3 | ✅ 已确认（下载后）：`aishell3_16khz/{speaker_id}/*.wav`（无 .trn）+ 根目录 jsonl（文本在 audio 字段）。`load_aishell3_speaker` 已改读 jsonl + `--text_jsonl` 参数 | modelscope `cqliushuai/aishell3-dataset` |
| T4 | ref_mels 加载层（见第4节 gap） | — |

### 3.2 `scripts/train_qwen3_tts_lora.py`（训练脚本）
| TODO | 做什么 | 对照源 |
|---|---|---|
| L1 | ✅ 已确认（probe）：`Qwen3TTSForConditionalGeneration` 在 `qwen_tts.core.models.modeling_qwen3_tts`。已填 `load_model` | probe |
| L2 | ✅ 已确认（probe）：`AutoConfig` 不认 qwen3_tts → 显式传 `Qwen3TTSConfig.from_pretrained`（方案A）。已填 | probe + `transformers.py:283` |
| L3 | ✅ 已确认（probe）：文本 tokenizer 是标准 Qwen3 文本 tokenizer（模型目录有 tokenizer_config.json），用 `AutoTokenizer.from_pretrained`。已填 | probe |
| L4 | ✅ 已确认：`CosineWarmupScheduler` 在 `twinkle.module.scheduler`，`construct_class` 能解析字符串。已填 | `twinkle/src/twinkle/module/scheduler/` |
| L5 | `model.forward` override 的 processor 路径：对「已 collate 2D batch」行为（对应 model.py TODO #5）。若报错→set `Qwen3TTSInputProcessor` 或 model.py 跳过 processor | `transformers.py:557-563` + model.py TODO #5 |
| L6 | AdamW（本脚本，skill TTS 经验 wd0.1）vs Muon（baseline 已验 NPU 可用）：AdamW 在 NPU 实测 | fsdp2.py:100 + skill §5 |

### 3.3 `tts/qwen3/model.py`（选项A 子类，详见该文件 docstring）
| TODO | 状态 |
|---|---|
| M1 `@remote_function` override 生效 | 容器实测 |
| M2 `forward_only` 不 override（eval 留 WER/SIM 阶段） | 已决策，非 bug |
| M3 `loss_value=0` 与 aux_loss 等价 | ✅ 已确认（读 transformers.py:624） |
| M4 PeftModel/FSDP 代理访问 talker/speaker_encoder/code_predictor | 容器实测 |
| M5 processor 对已 collate 2D batch 行为 | 容器实测（同 L5） |
| M6 跳过 postprocess_sp/unpack_packed（单卡无 SP） | 容器实测 |

## 4. 关键 gap：ref_mels 加载层

**问题**：`multi_codebook.py` 的 Processor 把 `ref_audio` 路径存进 `user_data`；
`template.py:60` 注释明说"mel 提取留 W-5"（未做），且该字段不进白名单会被 InputProcessor 丢弃；
`collate_qwen3_batch`(data.py:196) 的 `ref_mels` 是**外部传入参数**。所以走 Twinkle Dataset
标准链路（Processor→Template→InputProcessor→DataLoader）时，**无 ref_mels 注入层**。

**本会话绕开**：训练脚本用自定义 torch Dataset（`TTSManifestDataset`）内联
`np.load(ref_mel_path)` + `collate_qwen3_batch(ref_mels=...)`，跳过 Twinkle Dataset 抽象先跑通流程。

**迁回 Twinkle Dataset（后续，可选）**：在 `Qwen3TTSTemplate._encode_messages` 读
`ref_mel_path` → `np.load` → `sample['ref_mels']`，并把 `ref_mels` 加进
`Qwen3TTSInputProcessor` 白名单 + `padding_map`。数据流水线脚本已预存 `ref_mel_path`
（manifest 字段），迁回时直接用。

## 5. 进容器执行顺序（下个会话）

1. **进容器**（`docs/07` 启动命令），`ASCEND_RT_VISIBLE_DEVICES=7`（卡6 已占，训练/推理轮换）
2. **探明 qwen_tts 包 API**（一次小脚本 print 签名）：`Qwen3TTSTokenizer.encode` /
   `Qwen3TTSForConditionalGeneration` 导入路径 / `extract_mels`——对照 `docs/05` §2/§3
3. **下 AISHELL-3**（或先用合成 audio_codes 验训练循环，真实数据后补）
4. **填 prepare_aishell3.py 的 T1/T2/T3** → 跑出 `train.jsonl`/`dev.jsonl`
5. **填 train_qwen3_tts_lora.py 的 L1/L2/L3/L4** → `python train_qwen3_tts_lora.py --max_steps 2`
   看 loss 不崩（smoke-test）
6. **若 L5/M5（processor）报错** → set `Qwen3TTSInputProcessor` 或改 model.py 跳过 processor
7. **点亮 M1/M4/M6** 后 → 全量小步 SFT 看 loss 降（lr1e-4/wd0.1/cosine/epochs3起）
8. **评估**：WER（内容准）+ SIM（音色像）→ 对照零样本基线（留阶段3）
9. **部署**：卡7 vllm-omni（与训练轮换）

## 6. 待解项（非阻塞，沿用新会话必读）
- FLA fast path fallback torch（kernelize，非阻塞）
- 系统 sox 缺失（`yum install sox`，本脚本用 torchaudio/soundfile 已绕开）
- issue #279 评论 + Qwen3-TTS off-by-one bugfix PR：等 LoRA SFT + WER/SIM 实证再提交

## 7. 与新会话必读衔接
本会话产出后，**下一会话起点 = 进容器点亮 §3 TODO + 实测训练**（不再是"LoRA SFT 开发"，
开发已完成，进实测阶段）。`新会话必读.md` 第8节已更新指向本文件。
