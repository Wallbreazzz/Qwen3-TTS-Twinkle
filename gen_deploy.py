"""生成 w5_deploy.sh（自包含部署脚本）。

遍历 tts/ 下所有 .py，用 heredoc 内嵌进 bash 脚本（quoted heredoc 禁用变量展开，$ / 反引号安全；
content.rstrip('\\n') 让 cat 还原时恰好补回每行 \\n，100% 还原含尾换行的文件）。

tts/ 有改动时重跑本脚本即可更新 w5_deploy.sh。
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
TTS = os.path.join(ROOT, "tts")
SCRIPTS = os.path.join(ROOT, "scripts")   # 会话10 新增：训练/数据脚本也随 w5_deploy.sh 进容器

py_files = []
for src_dir in (TTS, SCRIPTS):
    if not os.path.isdir(src_dir):
        continue
    for dp, dirs, fs in os.walk(src_dir):
        dirs.sort()
        for f in sorted(fs):
            if f.endswith(".py"):
                py_files.append(os.path.join(dp, f))

header = r'''#!/bin/bash
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

'''

footer = r'''
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
'''

blocks = []
for p in py_files:
    rel = os.path.relpath(p, ROOT).replace("\\", "/")
    content = open(p, encoding="utf-8").read()
    stripped = content.rstrip("\n")
    blocks.append(f'mkdir -p "$(dirname "{rel}")"\ncat > "{rel}" <<\'PYEOF\'\n{stripped}\nPYEOF\n')

out = header + "".join(blocks) + footer
out_path = os.path.join(ROOT, "w5_deploy.sh")
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(out)
print(f"生成 {out_path}: {len(py_files)} 块, {os.path.getsize(out_path)} 字节")
