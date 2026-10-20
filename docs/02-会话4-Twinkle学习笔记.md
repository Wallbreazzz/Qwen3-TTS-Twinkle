# 会话 4 学习笔记：Twinkle 框架深入 + TTS 训练组件方案

> ⚠️ **时效性提示**：本文件是会话4 的阶段性认识，其中多码本组织/loss 相关假设**已被 `docs/05-W1-多码本组织调研.md`（权威）修正**（如 2 段式架构、codec id 默认值、ref_mels 形状等）。TTS 训练组件开发请以 `docs/05` 为准；本文件仅"codec 不训原因 / 微调能力边界（§5）/ NPU 需求拆分"等分析仍有独立参考价值。

> 本文件固化会话 4 的关键分析与产出，防止后续会话重复造轮子。
> 术语见 `docs/01-术语表与基础概念.md`；路线见 `docs/00-总体方案.md`；硬规则见 `AGENTS.md`。
> 本文件只放"会话 4 新增的深入分析"，不重复已有定义。

---

## 0. 会话 4 产出速览（防止重复造轮子）

| 产出 | 位置 | 用途 |
|---|---|---|
| Twinkle 仓库源码 | `twinkle/`（项目子目录） | 已 clone，可直接读源码、改源码提 PR |
| README 小白版 | `archive/readme_小白.md`（已归档） | 中文版 README 的逐段大白话翻译，新手入门 |
| 上游 issue #279 | https://github.com/modelscope/twinkle/issues/279 | TTS 训练支持方案，maintainer 已 review（见本文件第 4.3 节） |
| issue 草稿 | `archive/.issue_draft.md`（已归档） | 提交时的 body 留档，要改用 `gh issue edit 279` |
| 文本 vs TTS 数据流对照 | 本文件第 2 节 | 含简化词表数值示例，理解多码本训练格式的关键 |
| codec 不训练原因 | 本文件第 3 节 | 4 条技术理由，回答"为什么不训 vocoder" |
| 工作项 NPU 需求拆分 | 本文件第 4 节 | 哪些工作项要去机房、哪些在家做 |
| maintainer review 摘要 | 本文件第 4.3 节 | issue #279 的 4 点确认 + 多 LoRA 约束 + 模式2 选择 |
| 微调能力边界 | 本文件第 5 节 | 微调能改什么/不能改什么，指导数据采集 |

---

## 1. Twinkle 框架认知（基于真实源码 + README）

### 1.1 关键事实
- Twinkle 仓库已 clone 到 `twinkle/`，含 `cookbook/`（示例脚本集）、`src/`（源码）、`README.md`（英文，426 行，最新最全）、`README_ZH.md`（中文，404 行，是英文版翻译但有少量滞后）。
- 中文版与英文版有不同步：教程表 cookbook 路径对不上（如"EP FSDP2 LoRA 微调"中文版写 `ep_fsdp_qwen3_moe.py`，英文版写 `ep_fsdp2_lora_qwen3_5_moe.py`）。**以仓库实际文件名为准**。
- 中文版独有"算法接口外露便于调试"那句（英文版没有），强调"同一份代码到处跑 + 过程透明"——这点对**本项目很关键**：要改训练循环接入 TTS 多码本 loss，过程外露意味着改起来不费劲。

### 1.2 本项目最该看懂的两份 cookbook
- `cookbook/transformers/fsdp2.py`：文本 SFT 示例。**阶段 0 在文本 Qwen3 上跑通它**验证 Twinkle+昇腾可用。
- `cookbook/mm/fsdp2.py`：多模态 SFT 示例。**本项目要仿它的 `LatexOCRProcessor(Preprocessor)` 范式**写 TTS 的 `TTSMultiCodebookProcessor`。关键数据结构是 `Trajectory(messages=[Message(role, content, images=[])])`，TTS 版要把 `images` 换成 `audio_codes` 之类。

### 1.3 训练循环的关键 API（来自 `fsdp2.py` 真实代码）
```python
# 数据流
Dataset(DatasetMeta(...)) → set_template(...) → map(Preprocessor子类) → encode() → DataLoader
# 模型
TransformersModel(model_id=...) → kernelize(model) → add_adapter_to_model('default', LoraConfig, gradient_accumulation_steps=N)
# 配置
set_optimizer(...) → set_lr_scheduler(...) → set_loss(...)
# 循环（外露）
for batch in dataloader:
    model.forward_backward(inputs=batch, task='causal_lm')  # task 决定 loss 走哪条
    model.clip_grad_and_step()
    metric = model.calculate_metric(is_training=True)
model.save('last-checkpoint', adapter_name='default')
```
> 关键：`task='causal_lm'` 是文本单码本 CE。TTS 要新增 `task='multi_codebook_lm'` 走多码本 CE（见 issue #279 W4）。

---

## 2. 文本 vs TTS 训练数据流对照（含简化数值示例）

> 用极小词表演示原理，词表是简化的但数据流和数值逻辑与真实 Qwen3 完全一样。

### 2.1 简化词表
- **文本词表** V_text=20（真实 Qwen3 是 15 万+）：`0=<pad>, 1=<|im_start|>, 2=<|im_end|>, 3=user, 4=assistant, 5=你, 6=是, 7=谁, 8=我, 9=叫, 10=小, 11=明, 12=好, 13=，, 14=吗, 15=是的, 16=很, 17=高, 18=兴, 19=<bos>`
- **音频码本词表** V_audio=10（每码本 10 个 token，0-9），简化到 **2 个码本**（真实 Qwen3-TTS 是 16 个）。

### 2.2 文本 SFT 数据流（以"你是谁 → 我叫小明"为例）

| 步骤 | 模块 | 数据变成 |
|---|---|---|
| 0 原始数据 | DatasetMeta | `{"question":"你是谁","answer":"我是{model_name}"}` |
| 1 加载 | Dataset | 同上 |
| 2 绑模板 | Template | 数据不变，绑 ChatML 格式 |
| 3 预处理 | **Preprocessor**（`SelfCognitionProcessor`） | `Trajectory(messages=[Message('user','你是谁'), Message('assistant','我叫小明')])` |
| 4 编码 | Template+Tokenizer | `input_ids=[1,3,5,6,7,2,1,4,8,9,10,11,2]`（13 个 token）<br>`labels=[-100×8, 8,9,10,11,2]`（user 部分置 -100，只学 assistant） |
| 5 切批 | DataLoader(batch=2) | `input_ids` shape=[2,13]，`labels` shape=[2,13]，`attention_mask` shape=[2,13] |
| 6 模型 | TransformersModel+LoRA | A shape=[2,4] 全 0 初始化，B shape=[4,2] 随机小值；补丁初始=0 |
| 7 前向反向 | forward_backward(task='causal_lm') | logits shape=[2,13,20]；位置 8 label=8(我)，p[8]=0.25 → loss=-log(0.25)≈1.39 |
| 8 更新 | clip_grad_and_step | 梯度裁剪+Muon/AdamW 更新 LoRA A/B |
| 9 保存 | save('last-checkpoint') | adapter.bin（几 MB，真实几十 MB） |

### 2.3 TTS SFT 数据流（以 text="你好"+目标音频 2 帧+参考音频 2 帧为例）

⚠️ 标 `[新增]` 的步骤是本项目要为 Twinkle 新增的组件，基于多模态 `LatexOCRProcessor` 范式合理推断。
> ⚠️ 此表为会话4 早期推断（"ref 音频作前缀 / 复用 causal_lm / 16 路 CE"），**已由 W-1 调研修正**：实际是 speaker emb 注入 position 6、双通道 `[B,T,2]`、2 段 forward。权威数据流见 `docs/05` 第 4/5 节，本表仅留作思路演进留档。

| 步骤 | 模块 | 数据变成 |
|---|---|---|
| 0 原始数据 | manifest.jsonl | `{"text":"你好","audio_path":"/data/001.wav","ref_audio_path":"/data/ref.wav"}` |
| 1 离线预编码 | 官方 Qwen3TTSTokenizer（数据准备阶段，不在训练脚本） | `{"text":"你好","audio_codes":[[3,7],[5,2]],"ref_audio_codes":[[1,4],[6,8]]}`（shape=[2帧,2码本]） |
| 2 加载 | Dataset | 同上 |
| 3 绑模板 | `[新增]` Qwen3TTSTemplate | 数据不变 |
| 4 预处理 | `[新增]` **TTSMultiCodebookProcessor** | `Trajectory(messages=[Message('user','你好'),Message('assistant','<目标占位>')], user_data=[('audio_codes',pack_value(target_codes)),('ref_audio_codes',pack_value(ref_codes))])`（maintainer 确认：audio_codes 放 `user_data`，不改 Message） |
| 5 编码 | `[新增]` encode 逻辑 | `input_ids=[19,5,12,1,6,3]` shape=[1,6]<br>`labels=[[[5,-100],[12,-100],[-100,-100],[-100,-100],[3,7],[5,2]]]` shape=[1,6,**2**]（文本+参考部分置 -100） |
| 6 切批 | DataLoader(batch=1,grad_accum=16) | 单卡小 batch+梯度累积 |
| 7 模型 | TransformersModel+LoRA | LoRA 挂 q/k/v 等（与文本完全相同，LoRA 接入位置与架构无关） |
| 8 前向反向 | forward_backward(task='causal_lm')（**复用现有 task**，maintainer 确认；多码本差异由新 Loss 处理） | logits shape=[1,6,10]×**2 个 head**；位置 4 码本1 label=3 p=0.30 loss=1.20，码本2 label=7 p=0.35 loss=1.05 |
| 9 更新 | clip_grad_and_step | 同文本 |
| 10 保存 | save('best-by-wer') | **按 val WER 选 best**（非 last，案例1 反面教训） |

### 2.4 核心结论（一句话）
> ⚠️ 本结论为会话4 早期推断，已由 W-1（`docs/05`）修正：区别不止 Preprocessor/Loss，还有 forward 编排。

TTS 训练和文本训练在"框架流程"上**基本**相通（Dataset→Template→Preprocessor→encode→DataLoader→TransformersModel→LoRA→forward_backward→save），但 Qwen3-TTS 的区别有 **3 处**：① Preprocessor（产出双通道多码本 InputFeature，非 `Trajectory+user_data`）；② Loss（2 段式 `CE(码本0)+0.3×CE(码本1-15)`，非 16 路 CE）；③ **forward 编排**（speaker_encoder 前置 + 手动组装 inputs_embeds + 2 段 forward talker→sub_talker）——这 3 处即本项目要为 Twinkle 新增的部分，详见 `docs/05` 第 4-6 节。

### 2.5 shape 对照速查
> ⚠️ 本节为会话4 早期推断，已由 W-1 调研（`docs/05`）修正：Qwen3-TTS **不是 16 路并行 CE**，而是 **2 段式**（Talker 码本0 + Code predictor 码本1-15）。下表为修正版。

| | 文本（现成） | Qwen3-TTS（实际，见 `docs/05`） |
|---|---|---|
| input_ids | [B, T] | **[B, T, 2]**（双通道：文本 + codec0）+ `codec_ids[B,T,16]` |
| labels | [B, T]（1 条码本） | `codec_0_labels[B,T]`（码本0，-100）+ `codec_ids[B,T,16]`（码本1-15，喂 sub-talker） |
| logits | [B, T, V]（1 个 head） | Talker: `[B,T,V]`（码本0）；Code predictor: `[B,15,V]`（逐帧，15 head） |
| Loss | 单码本 CE | **`CE(码本0) + 0.3×CE(码本1-15)`**（2 段，非 16 路求和） |
| -100 位置 | user 部分 | 非目标音频帧（文本/头部/pad） |
| forward | 单次 `model(**inputs)` | **2 段**：talker → `forward_sub_talker_finetune`（+ speaker_encoder 前置 + 手动组装 inputs_embeds） |

---

## 3. 为什么不训练 vocoder / codec（4 条技术理由）

> 这点 `docs/00-总体方案.md` 和 skill 里只隐含说"冻结 codec"，本节讲清原因，避免后续会话重复解释。

不是偷懒，是 TTS 微调的标准做法：

1. **codec 与音色无关，是通用语音编解码器**：codec 学的是"波形 ↔ 离散 token"的通用转换，跟谁的声音没关系。厂商用海量多说话人数据训好了，能还原任何人的声音，改它没意义。
2. **音色差异在 LM 层不在 codec 层**：目标音色本质是"LM 倾向生成哪些 audio_codes 组合"。codec 只是把 token 还原成波形的"解码器"，改 LM 的生成分布才影响音色，改 codec 没用。
3. **改 codec 风险大**：单卡 LoRA 微调 codec 容易训坏（重建质量下降，出金属音/噪声），且 codec 是共享的，一改影响所有说话人。
4. **行业共识**：要镜像的 OpenMOSS 官方 `train_lora.py` 也只 LoRA 微调 LM 部分，不碰 codec。

**类比**：codec 是"音频的 ZIP 解压器"，LM 是"写 ZIP 文件的人"。要让生成的 ZIP 内容像某人的风格，应该改写 ZIP 的人（LM），不是改解压器（codec）。

**关键认知**：不训 codec ≠ 不用 codec。恰恰相反——audio_codes 就是 codec 编码出来的，是训练的"目标 label"。新增的 Twinkle 组件要能"吃"这种多码本 label，现成组件吃不了。

**例外**（不在本项目范围）：目标音色声学特征跟 codec 训练数据差异极大（特殊方言/特殊发声）时才需 fine-tune codec，那是大厂级工作。

---

## 4. issue #279 工作项的 NPU 需求拆分（去机房规划）

> 防止后续会话重复分析"哪些工作项要 NPU"。issue #279 的 7 个工作项拆分如下。

| 工作项 | 写代码 | 验证方式 | 去机房？ |
|---|---|---|---|
| W1 调研多码本组织 | — | 读官方代码+跑几条样本看 tokenizer 行为 | ❌ 不需要（CPU/本地，编码几条样本 CPU 慢但够） |
| W2 设计 Processor 接口 | 纸上设计 | — | ❌ 不需要 |
| W3 实现 Processor+单测 | 写代码 | 单测是纯数据加工（tensor 拼接、-100 mask），不调用 codec/模型前向 | ❌ 不需要（CPU 完全够） |
| W4 实现 Loss+单测 | 写代码 | 单测是纯 tensor 运算（多码本 CE 数学），用极小随机 tensor 测 | ❌ 不需要（CPU 够；fused-CE 内核是优化项，后置） |
| W5 cookbook 示例 | 写代码 | **真正跑通一次 LoRA SFT 训练** | ✅ **需要**（前向/反向必须在 NPU） |
| W6 MOSS 适配 | 写代码 | **真正跑通训练** | ✅ **需要** |
| W7 文档 | 写文档 | — | ❌ 不需要 |

### 4.1 节奏建议（攒一批去机房一次）
- **在家/办公室做**：W1 → W2 → W3 → W4 → W7，外加 W5/W6 的**代码编写**部分（W5a/W6a）。W3/W4 单测在 CPU 上跑通验证逻辑无 bug。
- **去机房集中做**：W5b + W6b——真正跑训练验证。攒着一次性去机房跑通，避免来回跑。
- **W5/W6 可先 CPU smoke test**：用极小模型+极小数据，CPU 能跑通流程（验证代码没 bug，但慢）。真正验证"loss 在降、合成有声音"必须 NPU。

### 4.2 不在 issue 7 项里但也要去机房的事
本项目阶段 0 的"在文本 Qwen3 上跑通 `cookbook/transformers/fsdp2.py` 验证 Twinkle+昇腾可用"（见 `新会话必读.md` 第 8 节）——是本项目预备工作，不在这 7 个上游工作项里，但和 W5 一样需要 NPU。建议和 W5b 一起去机房跑。

### 4.3 maintainer review 摘要（issue #279 已确认）

maintainer（@tastelikefeet，collaborator）已 review，4 个不确定点全明确：

| 不确定点 | maintainer 回答 | 落地含义 |
|---|---|---|
| Message 加 audio_codes 字段？ | 用 `Trajectory.user_data` 放其他字段；Processor 做成基类让类似数据集继承复用 | 不改 Message，audio_codes 放 `user_data=[('audio_codes', pack_value(codes)), ...]`；W2 设计成基类 |
| 新 Loss 还是 CE 加模式？ | 可以是一个新 loss | W4 写新 `TTSMultiCodebookLoss` 类 ✅ |
| task='multi_codebook_lm' 合适吗？ | task 是针对模型结构有修改（embedding/reranker）的；只 loss 不同就**复用现有 task** | W4 **不新增 task**，复用 `causal_lm`，差异全在 Loss 层 |
| cookbook 放哪？ | "相对随意" | 选 `cookbook/tts/` ✅ |

**maintainer 新提的 2 点（不在原 issue 里）**：
1. **多 LoRA 兼容**：twinkle 支持 c/s 模式（多租户共享底座各挂 LoRA），新组件要兼容多 LoRA。→ W2/W4 设计时 Processor 不写死 `adapter_name='default'`，Loss 按传入 adapter 算。
2. **swift 在和 twinkle 底层架构融合**，9 月末稳定。两个时机选择，**本项目选模式2（先行合入）**：现在开始 W1-W4，9 月末重构如有适配再调整。理由：W1-W4 是上层组件（Preprocessor/Loss），底层重构（Dataset/Template）影响相对小；先行 PR 能让 maintainer 重构时主动兼容我的接口。

**关键 API 确认**（来自 `twinkle/src/twinkle/data_format/trajectory.py:15-19`）：
```python
class Trajectory(TypedDict, total=False):
    messages: List[Message]
    user_data: List[Tuple[str, str]]  # PyArrow-stable: (key, json.dumps(value))
    audios: Optional[List[Union[str, Any]]]  # 已有 audios 字段，但 user_data 更通用
```
- 用 `pack_value(value)` 编码（json.dumps），`user_data_get(items, key)` 解码。

---

## 5. 微调能力边界：能改什么/不能改什么

> 这点 docs/skill 都没明说，本节固化为数据采集和评估提供指导。

### 5.1 微调 talker（LM）能影响哪些维度
TTS 输出的所有"动态特征"——语速、语气、停顿、韵律、情感——**本质上都编码在 LM 输出的 audio token 序列里**。codec 只负责把 token 还原成波形，不知道语速/语气。所以微调 LM = 改变 token 分布 = **能影响所有这些维度，不只是音色**。

| 维度 | 由谁决定 | 微调 LM 能改吗 |
|---|---|---|
| 音色（声纹） | LM 的 token 分布 + codec 还原 | ✅ 能（最明显） |
| 语速 | LM 生成的帧数/序列长度 | ✅ 能 |
| 停顿 | LM 生成的静音 token 序列 | ✅ 能 |
| 韵律（语调起伏） | LM 的 token 分布 | ✅ 能 |
| 语气/情感 | LM 的 token 分布 | ✅ 能（但有前提） |

### 5.2 三个重要前提
1. **上限由训练数据决定**：LM 学的是你喂的数据分布。训练数据里有的语速/语气/停顿习惯才能学到；数据单一不会无中生有变得有语气。
   - 类比：LM 是"模仿演员"，听某人录音学会这个人的特征——但只学会录音里有的特征。
2. **微调是"模仿"不是"创造"**：next-token CE 让 LM 拟合训练数据 token 分布。能学会目标人的习惯，不能凭空生成数据里没有的情感。
3. **音色提升最明显，其他维度难量化**：音色相似度（SIM）有自动指标；语速/停顿/语气像不像目标人，**没有好的自动指标**，要靠人听 + 间接看 WER。

### 5.3 微调做不到的：精确控制
"让这句话用开心语气说""让第 3 个字重读"这种**细粒度可控控制**，单纯 SFT 给不了——需要额外控制信号（emotion tag / prosody prompt），那是可控 TTS 的范畴，**超出本项目**。本项目做的是"让模型整体听起来像目标人"，不是"指令级控制语气"。

### 5.4 对本项目的实践含义
1. **训练数据要尽量多样**：5~15 分钟录音要覆盖目标人的不同句型、不同语速、不同语气——只录平淡朗读，拟人化上限就低。这是 `tts-data-pipeline` skill 强调"数据质量 >> 数量"的深层原因。
2. **评估不能只盯 SIM**：还要听语速/停顿是否自然像目标人（人耳听 + WER 间接反映清晰度/语速）。
3. **想未来做可控语气**：要加 emotion/prosody 控制信号，那是另一个项目——但底座选 Qwen3-TTS 是对的，AR-LM 架构后续加控制信号比非 AR 容易。

---

## 6. 关键认知汇总（一句话总结，便于记忆）

1. **Twinkle 现无 TTS 支持**，要新增 `TTSMultiCodebookProcessor`（Preprocessor）+ `TTSMultiCodebookLoss`（Loss）两个组件，且 W-1 发现**还需自定义 forward 编排**（见 `docs/05` 第 6 节）——这是 issue #279 的核心。
2. **TTS 和文本训练在框架流程上"基本"相通**，但区别**不止** Preprocessor 和 Loss：Qwen3-TTS 官方循环手动做 speaker_encoder 前置 + inputs_embeds 组装 + 2 段 forward（talker→sub_talker），**超出** Twinkle 标准 `forward_backward(task='causal_lm')` 单次 forward 范围（见 `docs/05` 第 5/6 节）。
3. **codec 不训练**（4 条理由），但 audio_codes 是 codec 编码出来的训练 label。
4. **工作项 W3/W4 在家写+CPU 测，W5/W6 去机房跑真训练**。
5. **微调能改音色+语速+语气+停顿+韵律**（都编码在 token 序列里），但上限由数据多样性决定，精确控制超范围。
6. **maintainer 已 review issue #279**：4 点确认（audio_codes 放 `user_data` / 新 Loss 类 / **复用 `causal_lm` task 不新增** / cookbook 放 `cookbook/tts/`）+ 多 LoRA 兼容约束 + 选模式2 先行合入（swift 融合 9 月末稳定，届时适配）。⚠️ 但 W-1 调研发现 Qwen3-TTS 的 2 段 forward + speaker 前置**可能需要自定义 task/Patch**（与"复用 causal_lm"有出入），W-2 验证后定；按用户指示 issue 范围由开发成果定义。详见第 4.3 节 + `docs/05`。
