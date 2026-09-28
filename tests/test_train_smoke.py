"""A few CPU steps of stage 1 and stage 2 on the tiny model + generated toy data."""
import json
import math
import os

import pytest
import torch
from safetensors.torch import load_file

from laya_vision.train.loop import load_config, seed_everything, train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(ROOT, "configs", "smoke.yaml")


def _cfg(tmp, tiny_laya_dir, *overrides, stage_defaults=None):
    return load_config(SMOKE, ["output_dir=%s" % tmp, "base=%s" % tiny_laya_dir,
                               "data.synthetic.dir=%s" % os.path.join(os.path.dirname(tmp), "data"),
                               "data.synthetic.n_train=16", "data.synthetic.n_eval=8",
                               "train.eval_every=1000", "train.save_every=2"] + list(overrides), stage_defaults)


def _metrics(out):
    with open(os.path.join(out, "metrics.jsonl")) as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def stage1(tmp_path_factory, tiny_laya_dir):
    out = str(tmp_path_factory.mktemp("run") / "stage1")
    cfg = _cfg(out, tiny_laya_dir, "train.max_steps=3")
    # the trainer seeds before building, so this reproduces its initial projector
    from laya_vision.checkpoint import init_from_laya
    from laya_vision.config import VisionConfig
    from laya_vision.testing import tiny_siglip_config

    seed_everything(cfg["seed"])
    init, _, _ = init_from_laya(tiny_laya_dir, VisionConfig.from_dict(cfg["vision"]), tower_config=tiny_siglip_config())
    res = train(cfg)
    return out, res, {k: v.clone() for k, v in init.state_dict().items()}


def test_stage1_trains_projector_only(stage1, tiny_laya_dir):
    out, res, init = stage1
    assert res["step"] == 3
    train_logs = [m for m in _metrics(out) if m["event"] == "train"]
    assert train_logs and all(math.isfinite(m["loss"]) for m in train_logs)

    from laya_vision.checkpoint import load_checkpoint

    model, tok, cfg, vcfg = load_checkpoint(os.path.join(out, "final"))
    sd = model.state_dict()
    # projector moved (fp16 checkpoint: compare with a tolerance well above rounding)
    assert any((sd[k].float() - init[k].float()).abs().max() > 1e-3 for k in init if k.startswith("projector."))
    # Laya weights unchanged: saved model.safetensors == original tiny Laya (up to fp16 rounding)
    base = load_file(os.path.join(tiny_laya_dir, "model.safetensors"))
    new = load_file(os.path.join(out, "final", "model.safetensors"))
    assert set(base) == set(new)
    for k in base:
        if base[k].is_floating_point():
            assert torch.allclose(base[k].half().float(), new[k].float(), atol=1e-3), k
    # tower unchanged too
    for k in init:
        if k.startswith("vision."):
            assert torch.allclose(init[k].half().float(), sd[k].float(), atol=1e-3), k


def test_stage1_rolling_checkpoint_and_resume(stage1, tiny_laya_dir):
    out, _, _ = stage1
    latest = os.path.join(out, "checkpoint_latest")
    assert os.path.exists(os.path.join(latest, "trainer_state.pt"))
    assert os.path.exists(os.path.join(out, "final", "vision.safetensors"))
    # continue the same run to 5 steps from checkpoint_latest
    res = train(_cfg(out, tiny_laya_dir, "train.max_steps=5", "resume=auto"))
    assert res["step"] == 5


def test_stage2_with_lora_saves_plain_laya(stage1, tmp_path, tiny_laya_dir):
    s1, _, _ = stage1
    from laya_vision.train.stage2 import STAGE2_DEFAULTS

    out = str(tmp_path / "stage2")
    cfg = _cfg(out, tiny_laya_dir, "init_from=%s" % os.path.join(s1, "final"), "train.max_steps=2",
               "train.lr={projector: 1.0e-4, encoder: 1.0e-3, head: 1.0e-4, vision: 0.0}",
               "train.lora.enabled=true", stage_defaults=STAGE2_DEFAULTS)
    cfg["stage"] = 2
    res = train(cfg)
    assert res["step"] == 2
    assert all(math.isfinite(m["loss"]) for m in _metrics(out) if m["event"] == "train")

    before = load_file(os.path.join(s1, "final", "model.safetensors"))
    after = load_file(os.path.join(out, "final", "model.safetensors"))
    assert set(before) == set(after)  # LoRA merged: plain Laya keys
    assert any((before[k].float() - after[k].float()).abs().max() > 0 for k in before if k.startswith("encoder."))

    import laya

    agent = laya.Agent(os.path.join(out, "final"), device="cpu")  # stock Laya still loads it
    ans = agent.predict("a red picture", {"q": {"type": "noul", "instructions": "Is it red?"}})
    assert "q" in ans["answers"]


def test_lora_merge_matches_adapter_forward():
    from laya_vision.train.lora import apply_lora, merged_copy

    torch.manual_seed(0)
    net = torch.nn.Sequential()
    net.add_module("Wqkv", torch.nn.Linear(8, 8))
    net.add_module("Wo", torch.nn.Linear(8, 8))
    params = apply_lora(net, r=4, alpha=8, dropout=0.0)
    for p in params:
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randn(3, 8)
    merged = merged_copy(net)
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    assert torch.allclose(net(x), merged(x), atol=1e-5)
