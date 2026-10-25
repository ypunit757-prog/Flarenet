import torch

from flarenet.core.constants import N_CHANNELS, N_CLASSES
from flarenet.models.flarenet import ChannelDropout, FlareNet, count_parameters
from flarenet.models.losses import FocalBCE, LabelSmoothedCE, class_weights


def test_forward_shapes():
    m = FlareNet()
    now, fc, attn = m(torch.randn(4, N_CHANNELS, 90))
    assert now.shape == (4, N_CLASSES) and fc.shape == (4,) and attn.shape == (4, 45)
    assert torch.allclose(attn.sum(1), torch.ones(4), atol=1e-5)


def test_model_is_small():
    assert 10_000 < count_parameters(FlareNet()) < 200_000


def test_backward_gives_finite_grads():
    m = FlareNet()
    now, fc, _ = m(torch.randn(8, N_CHANNELS, 90))
    loss = now.logsumexp(1).mean() + fc.mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_eval_mode_is_deterministic():
    m = FlareNet().eval()
    x = torch.randn(3, N_CHANNELS, 90)
    a, b = m(x), m(x)
    assert all(torch.equal(u, v) for u, v in zip(a, b, strict=True))


def test_channel_dropout_noop_in_eval_and_scales_in_train():
    d = ChannelDropout(0.5).eval()
    x = torch.ones(2, 8, 10)
    assert torch.equal(d(x), x)
    d.train()
    out = d(x)
    assert set(out.unique().tolist()) <= {0.0, 2.0}


def test_losses():
    logits = torch.zeros(4)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert FocalBCE(2.0)(logits, targets) > 0
    ce = LabelSmoothedCE(5, 0.1, class_weights(torch.tensor([0, 0, 1, 2]).numpy(), 5))
    assert ce(torch.randn(4, 5), torch.tensor([0, 1, 2, 3])) > 0
    w = class_weights(torch.tensor([0, 0, 0, 1]).numpy(), 5, cap=2.0)
    assert w.max() <= 2.0 and w.shape == (5,)
