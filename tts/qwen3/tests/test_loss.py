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
