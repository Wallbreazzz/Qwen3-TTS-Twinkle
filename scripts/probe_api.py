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
