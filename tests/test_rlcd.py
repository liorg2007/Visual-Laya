import pytest
import torch
from laya.common import proper_reward

from laya_vision.train.rlcd import rlcd_loss, sigma_at


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    B, K = 6, 5
    ks = torch.tensor([2, 5, 3, 4, 2, 5])
    mask = torch.arange(K)[None] < ks[:, None]
    qtype = torch.tensor([2, 0, 1, 0, 2, 1])
    raw = torch.randn(B, K, generator=g) * 2
    t = torch.rand(B, K, generator=g) * mask
    t[0] = torch.tensor([0.0, 1.0, 0, 0, 0])  # one-hot row
    target = t / t.sum(-1, keepdim=True)
    return raw, target, mask, qtype


def _notebook_loss(logits, target, mask, qtype, sigma, GROUP_SIZE=4, GRAD_ACCUM=1):
    # verbatim from the notebook's train_ddp.py (device -> cpu, act term dropped)
    device = logits.device
    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM


@pytest.mark.parametrize("sigma", [0.4, 0.25, 0.1])
def test_matches_notebook(sigma):
    raw, target, mask, qtype = _batch()
    logits = raw.masked_fill(~mask, -1e4)  # as DecisionModel returns them
    torch.manual_seed(123)
    ref = _notebook_loss(logits, target, mask, qtype, sigma)
    torch.manual_seed(123)
    loss, stats = rlcd_loss(logits, target, mask, qtype, sigma)
    assert torch.allclose(loss, ref, atol=1e-6, rtol=1e-6)
    assert set(stats) >= {"loss", "loss_rl", "loss_ce", "reward", "sigma"}


def test_gradient_flows_and_masked_options_get_none():
    raw, target, mask, qtype = _batch(1)
    raw = raw.clone().requires_grad_(True)
    logits = raw.masked_fill(~mask, -1e4)
    torch.manual_seed(0)
    loss, _ = rlcd_loss(logits, target, mask, qtype, 0.3)
    loss.backward()
    assert torch.isfinite(loss)
    assert raw.grad[mask].abs().sum() > 0
    assert torch.all(raw.grad[~mask] == 0)


def test_masked_logits_without_fill_still_get_no_gradient():
    raw, target, mask, qtype = _batch(2)
    raw = raw.clone().requires_grad_(True)
    torch.manual_seed(0)
    loss, _ = rlcd_loss(raw, target, mask, qtype, 0.3)
    loss.backward()
    assert torch.all(raw.grad[~mask] == 0)


def test_ce_only_and_generator():
    raw, target, mask, qtype = _batch(3)
    logits = raw.masked_fill(~mask, -1e4)
    a, _ = rlcd_loss(logits, target, mask, qtype, 0.2, generator=torch.Generator().manual_seed(5))
    b, _ = rlcd_loss(logits, target, mask, qtype, 0.2, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
    _, s = rlcd_loss(logits, target, mask, qtype, 0.2, ce_weight=0.0)
    assert s["loss"] == pytest.approx(s["loss_rl"])


def test_sigma_schedule():
    assert sigma_at(0.0) == pytest.approx(0.4)
    assert sigma_at(1.0) == pytest.approx(0.1)
    assert sigma_at(0.5) == pytest.approx(0.25)
    assert sigma_at(2.0) == pytest.approx(0.1)
