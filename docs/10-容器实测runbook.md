# 容器实测 runbook：进卡7 NPU 跑通 Qwen3-TTS LoRA SFT（会话10 产出）

> 本文件是"进远程服务器 → 用 NPU 验证 LoRA SFT 跑通"的**可操作步骤清单**。每步带命令、
> 预期结果、出错对照。照着顺序跑即可。
> 配合：`docs/07`（环境/容器命令）、`docs/08`（TODO 清单）、`w5_deploy.sh`（铺码）、
> `docs/05`（API 权威）。术语见 `docs/01`；硬规则见 `AGENTS.md`。

## 0. 前置（本会话已完成）
- 已写好 cookbook 骨架：`scripts/prepare_aishell3.py`（数据）、`scripts/train_qwen3_tts_lora.py`（训练）、
  `tts/qwen3/model.py`（选项A 编排），全是**骨架 + TODO**，待 NPU 实测填 API 后跑通。
- `w5_deploy.sh` 已更新（本会话重跑 `gen_deploy.py`），**一次铺 `tts/` + `scripts/` 全部** + 回归单测。
- 待你：登远程服务器（卡7，训练/推理轮换）。

---

## 步骤 1 — 起训练容器（卡7，docs/07 §4）

先在宿主确认卡7 空闲（vllm-omni 推理没在跑，训练/推理轮换）：
```bash
npu-smi info | head -20    # 看卡7 是否空闲
```
起容器（镜像自带 CANN9.1.0 + 挂宿主 driver，docs/07 §4 实测命令）：
```bash
docker run -itd -u 0 --name h00875519_twinkle_train \
  --ipc=host --privileged --network=mynet1400 --cpus=24 --memory=200g \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache -v /home/:/home/ \
  -e ASCEND_RT_VISIBLE_DEVICES=7 \
  quay.nju.edu.cn/ascend/ms-swift:v4.5.2-cann9.1.0-torch_npu2.10.0.post2-910b-openeuler24.03-py3.12 bash
```
- **预期**：容器起，`docker ps` 见 `h00875519_twinkle_train`。
- **出错**：卡7 占用 → 等 vllm-omni 推理停（轮换）；镜像没有 → `docker pull` 那个镜像名。

---

## 步骤 2 — 进容器 + 验证 NPU（docs/07 §5.1）
```bash
docker exec -it h00875519_twinkle_train bash
# 容器内：
python -c "import torch,torch_npu;print('npu avail',torch.npu.is_available(),'count',torch.npu.device_count())"
npu-smi info | head -5     # 确认看到卡7
ls /usr/local/Ascend/      # 应见 cann-9.1.0
```
- **预期**：`npu avail True count 1`（ASCEND_RT_VISIBLE_DEVICES=7 限定单卡）。
- **出错**：`import torch_npu` 失败 → driver 没挂对，查 `-v /usr/local/Ascend/driver/...`。

---

## 步骤 3 — 装训练栈（docs/07 §5.2，会话9 已验证）
```bash
pip config set global.index-url https://mirrors.cloud.aliyuncs.com/pypi/simple
pip install twinkle-kit peft flash-linear-attention
pip install -U qwen-tts       # 拉全依赖：transformers==4.57.3 + accelerate + sox + torchaudio + librosa
python -c "import twinkle,qwen_tts;print('共存 OK')"
pip show transformers | grep Version    # 必须是 4.57.3（qwen-tts 钉版本）
```
- **预期**：`共存 OK` + `Version: 4.57.3`。
- **出错**：transformers 不是 4.57.3 → `pip install transformers==4.57.3` 强钉；FLA 编译失败 → 先不管（步骤 11 出错对照有 fallback）。

---

## 步骤 4 — 同步代码（w5_deploy.sh 一次铺 tts/ + scripts/）
从本机把 `w5_deploy.sh` + `w5_verify_tts.py` 拷进容器（scp/cp/heredoc 任选），然后：
```bash
bash w5_deploy.sh /workspace
```
- **预期**（段 0-3）：
  - 段0 环境检查全 OK
  - 段1 铺好 `tts/`（16 文件）+ `scripts/`（2 文件：prepare_aishell3.py / train_qwen3_tts_lora.py）
  - 段2 4 个回归单测全过（`test_template` real-tokenizer 集成测在权重未下时 SKIP 正常）
  - 段3 `Qwen3TTSModel` 等全部 import 通
- **出错**：段0 FAIL → 回步骤 3 装栈；单测 FAIL → 查 docs/08 §3.3 model.py TODO。

---

## 步骤 5 — 探明 qwen_tts API（点亮 TODO 的关键，本机无法验）

这是进容器**最该先做**的——已落盘 `scripts/probe_api.py`（步骤4 `w5_deploy.sh` 铺好），探 T1/T2/L1/L2/L3/L4 全部 TODO，对照 `docs/05` §2.1/§3.3 确认后填进 `prepare_aishell3.py`/`train_qwen3_tts_lora.py`。

```bash
python scripts/probe_api.py                                              # 探全（不含 encode 实跑）
python scripts/probe_api.py --tokenizer_path /path/to/Qwen3-TTS-Tokenizer-12Hz  # 加测 encode 签名（需下权重）
python scripts/probe_api.py --model_path /home/h00875519/Qwen3-TTS-12Hz-1.7B-Base
```
- **预期**：打印各模块可 import 的类名、extract_mel 签名、AutoConfig 是否认 qwen3_tts。
- **产出**：把确认的导入路径 + 签名填进 `scripts/prepare_aishell3.py`（T1/T2）+ `scripts/train_qwen3_tts_lora.py`（L1/L2/L3/L4）的 TODO。
- **对照**：`docs/05` §2.1 `Qwen3TTSTokenizer.encode(audios).audio_codes` → `[T,16]`；§3.3 `mel_dim=128/sample_rate=24000`。

---

## 步骤 6 — 下 AISHELL-3 单说话人（实测结构已确认）

数据集在 modelscope **`cqliushuai/aishell3-dataset`**（非 openslr，非 `speechbrain/aishell3`——后者不存在）。
按说话人目录组织（`aishell3_16khz/SSB0005/...`），**单说话人 ~71MB**，10GB 磁盘够下 100+ 个。

只下 SSB0005（wav）+ 全量 jsonl（**文本在 jsonl 的 audio 字段，SSB 目录下无 .trn**）：
```bash
modelscope download cqliushuai/aishell3-dataset \
  --repo-type dataset \
  --local-dir /home/h00875519/aishell3 \
  --include "aishell3_16khz/SSB0005/*" "aishell3_tts.jsonl"
```
- **参数**（实测 `modelscope download --help`）：`--include`（glob，可重复，**非** `--allow_patterns`）；
  repo_id 是 positional + `--repo-type dataset`（**非** `--dataset`）。
- **下载后确认**：
  ```bash
  ls /home/h00875519/aishell3/aishell3_16khz/SSB0005/ | head    # SSB0005xxxx.wav（无 .trn）
  du -sh /home/h00875519/aishell3/aishell3_16khz/SSB0005/       # ~71MB
  head -1 /home/h00875519/aishell3/aishell3_tts.jsonl           # {"audio":"./aishell3_16khz/SSBxxxx/xxx.wav","text":"..."}
  ```
- `prepare_aishell3.py` 的 `load_aishell3_speaker` **已适配此结构**（读 jsonl 按 speaker_id 过滤），
  传 `--aishell3_root /home/h00875519/aishell3 --text_jsonl aishell3_tts.jsonl`。

---

## 步骤 7 — 填 prepare_aishell3.py TODO → 跑出 manifest
基于步骤 5 探明的 API，填 `scripts/prepare_aishell3.py` 的：
- **T1** `Qwen3TTSTokenizer` 导入 + `from_pretrained` + `encode` 签名（audios 是 wav 张量？list？）
- **T2** 官方 `extract_mels` 路径（或确认 torchaudio kaldi fallback 格式一致）
- ✅ **T3 已确认并适配**：`aishell3_16khz/speaker_id/*.wav` + jsonl 文本（`load_aishell3_speaker` 已改读 jsonl）
```bash
python scripts/prepare_aishell3.py \
  --aishell3_root /home/h00875519/aishell3 --speaker_id SSB0005 \
  --text_jsonl aishell3_tts.jsonl \
  --output_dir data/qwen3_aishell3 \
  --speech_tokenizer_path /path/to/Qwen3-TTS-Tokenizer-12Hz
```
- **预期**：`data/qwen3_aishell3/train.jsonl` + `dev.jsonl` 生成，每行含
  `text/audio_codes[T,16]/ref_mel_path/language`；train≠dev 拆 15%。
- **出错**：codec id 查表超词表 → 确认 `Qwen3TTSDataConfig` 从 `model.config.talker_config` 读（w5_verify 做法）。

---

## 步骤 8 — w5_verify_tts.py 回归（确认前向 OK，会话9 已验）
```bash
python w5_verify_tts.py --model_path /home/h00875519/Qwen3-TTS-12Hz-1.7B-Base
```
- **预期**（会话9 数值基准）：`talker_loss ≈ 15.58`、`total ≈ 18.22`（shift=False）；
  `sub_talker_logits [N,15,V]`、`sub_talker_labels [N,15]` 形状对。
- **作用**：确认环境 + 前向编排 + loss 在真模型上 OK，再进训练循环。

---

## 步骤 9 — 填 train_qwen3_tts_lora.py TODO → smoke-test（核心闭环）
基于步骤 5 探明，填 `scripts/train_qwen3_tts_lora.py` 的：
- **L1** `Qwen3TTSForConditionalGeneration` 导入路径
- **L2** config 加载（AutoConfig 认就直接用；不认传 `Qwen3TTSConfig` 或 override `__init__`）
- **L3** tokenizer 获取（`model.tokenizer`？或 qwen_tts 的文本 tokenizer）
- **L4** scheduler 名（`'CosineWarmupScheduler'` 是否 Twinkle 支持，查 twinkle scheduler 注册表）
```bash
ASCEND_RT_VISIBLE_DEVICES=7 python scripts/train_qwen3_tts_lora.py \
  --model_path /home/h00875519/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl data/qwen3_aishell3/train.jsonl \
  --dev_jsonl   data/qwen3_aishell3/dev.jsonl \
  --output_dir  outputs/qwen3_tts_lora \
  --max_steps 2
```
- **预期**：2 步 `forward_backward` + `clip_grad_and_step` 不崩，`metric` 打印 loss 数值合理（非 nan）。
- **这是最小可验证闭环**——跑通= model.py override + LoRA 挂载 + 训练循环 + 数据侧 全链路 OK。
- **出错**：
  - processor 对已 collate 2D batch 报错（L5/M5）→ `model.set_processor(Qwen3TTSInputProcessor)` 或改 model.py 跳过 processor
  - `model.talker` 访问报错（M4 PeftModel 代理）→ 查 PeftModel 是否透传
  - kernelize 报错 → `export TWINKLE_TORCH_BASELINE=1` 跑纯 torch 基线

---

## 步骤 10 — 点亮 model.py 实测 TODO（smoke-test 跑通即间接验证大部分）
| TODO | 验证方式 |
|---|---|
| M1 `@remote_function` override 生效 | smoke-test 跑通=生效 |
| M2 `forward_only` 不 override | 已决策留 WER/SIM 阶段（eval 暂跳过） |
| M3 loss_value=0 等价 aux_loss | ✅ 已确认 |
| M4 PeftModel 代理访问 talker/speaker_encoder | smoke-test 前向不崩=透传 OK |
| M5 processor 对已 collate batch | 同步骤 9 的 L5 |
| M6 跳过 postprocess_sp（单卡） | smoke-test 不崩=可跳过 |

---

## 步骤 11 — 全量小步 SFT 看 loss 降（smoke-test 通过后）
```bash
python scripts/train_qwen3_tts_lora.py \
  --model_path /home/h00875519/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl data/qwen3_aishell3/train.jsonl \
  --output_dir outputs/qwen3_tts_lora \
  --epochs 3           # 去掉 --max_steps，跑全量
```
- **预期**：loss 下降（talker_loss + 0.3×sub_loss），`log_interval` 打印 metric。
- **超参**（已定，docs/09 §9 验证）：r=8/alpha=16/lr1e-4/wd0.1/cosine+warmup0.03/epochs3。
- **盯**：loss 不降反升=过拟合 → 降 epochs/lr，加 dropout（case1 教训）。

---

## 步骤 12 — 评估 WER/SIM（留阶段3）
- 合成语音 → ASR 算 **WER**（内容准）+ **说话人相似度 SIM**（音色像）。
- 对照零样本基线看收益；按 val WER 早停选 best checkpoint。
- **shift=False vs True 对比**：积累 WER/SIM 证据，为 Qwen3-TTS off-by-one bugfix PR 提数据。

---

## 步骤 13 — 部署（卡7 vllm-omni，训练/推理轮换）
- adapter 存 `outputs/qwen3_tts_lora/last-checkpoint`。
- 训练停 → 卡7 vllm-omni 容器挂 adapter 推理（加载 `vllm-omni-tts-integration` skill）。
- CANN 9.1.0 一致（训练 ms-swift 镜像 / 推理 vllm-omni 镜像都是 9.1.0），adapter 可跨容器即挂。

---

## 出错对照速查
| 症状 | 处理（详见 docs/07 §6 + docs/08）|
|---|---|
| `kernelize` 报算子未实现 | `export TWINKLE_TORCH_BASELINE=1` 跑纯 torch 基线 |
| FLA 编译失败 | 同上，先 `TWINKLE_TORCH_BASELINE=1` 跳过 |
| processor 对已 collate batch 报错 | `model.set_processor(Qwen3TTSInputProcessor)` 或 model.py 跳过 processor |
| `model.talker` 访问报错 | 查 PeftModel/FSDP getattr 透传 |
| 卡7 占用 | `npu-smi info` 确认 + 等 vllm-omni 推理停（轮换） |
| loss=nan | 确认 bf16、grad_clip、audio_codes 无异常值 |
| codec id 查表超词表 | `Qwen3TTSDataConfig` 从 `model.config.talker_config` 读真实值 |

## 里程碑
- **步骤 9 smoke-test 跑通** = LoRA SFT 全链路验证 OK（本会话 cookbook 串接成功）。
- **步骤 11 loss 下降** = 真实可用的微调。
- **步骤 12 WER/SIM 达标** = 可以交付 + 提 issue #279 评论 + off-by-one bugfix PR。
