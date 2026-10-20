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
