"""UNet forward-shape and conditioning contracts (bind.model).

The channel bookkeeping is spread over model.py + data.py + train.py:

    base_out = 4 if stars_two_head else 3
    out_ch   = base_out + (N_THERMO if predict_thermo else 0)
    in_ch    = out_ch + 1 (condition) + 3 (large_scale)

These tests pin that arithmetic and the redshift-conditioning behaviour with a
deliberately tiny UNet so they run in well under a second on CPU.
"""

import itertools

import numpy as np
import pytest
import torch

from bind.data import N_THERMO
from bind.model import UNet

RES = 32          # smaller than the production 128; only the conv stack cares
BATCH = 2
N_PARAMS = 35

# Tiny but structurally faithful: GroupNorm(32, ch) requires ch % 32 == 0, so
# base_ch must stay a multiple of 32.  attn_resolutions=() drops the (costly)
# per-level attention; the bottleneck attention block is unconditional.
TINY = dict(base_ch=32, ch_mult=(1, 2), n_blocks=1, emb_dim=64,
            attn_resolutions=(), dropout=0.0, n_params=N_PARAMS)


def _channels(stars_two_head, predict_thermo):
    out_ch = (4 if stars_two_head else 3) + (N_THERMO if predict_thermo else 0)
    return out_ch + 1 + 3, out_ch


def _build(stars_two_head, predict_thermo, condition_redshift, seed=0):
    torch.manual_seed(seed)
    in_ch, out_ch = _channels(stars_two_head, predict_thermo)
    net = UNet(in_ch=in_ch, out_ch=out_ch, condition_redshift=condition_redshift, **TINY)
    # out_conv is zero-initialized by design (identity velocity at init), which
    # would make every output trivially equal.  Give it real weights so the
    # conditioning tests can actually detect a difference.
    torch.nn.init.normal_(net.out_conv.weight, std=0.05)
    torch.nn.init.normal_(net.out_conv.bias, std=0.05)
    return net.eval(), in_ch, out_ch


def _inputs(in_ch, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, in_ch, RES, RES, generator=g)
    t = torch.rand(BATCH, generator=g)
    params = torch.rand(BATCH, N_PARAMS, generator=g)
    return x, t, params


@pytest.mark.parametrize(
    "stars_two_head,predict_thermo,condition_redshift",
    list(itertools.product([False, True], repeat=3)),
)
def test_forward_output_channels(stars_two_head, predict_thermo, condition_redshift):
    net, in_ch, out_ch = _build(stars_two_head, predict_thermo, condition_redshift)
    x, t, params = _inputs(in_ch)
    scale_factor = torch.full((BATCH,), 0.5) if condition_redshift else None

    with torch.no_grad():
        y = net(x, t, params, scale_factor=scale_factor)

    assert y.shape == (BATCH, out_ch, RES, RES)
    assert torch.isfinite(y).all()
    # Sanity on the arithmetic itself, independent of the model.
    assert in_ch == out_ch + 4
    assert out_ch == (4 if stars_two_head else 3) + (N_THERMO if predict_thermo else 0)


def test_redshift_model_defaults_missing_scale_factor_to_a_equals_one():
    """A condition_redshift model given scale_factor=None must behave as z=0
    (a = 1), not error and not silently drop the embedding."""
    net, in_ch, _ = _build(False, False, True, seed=2)
    x, t, params = _inputs(in_ch)

    with torch.no_grad():
        y_none = net(x, t, params, scale_factor=None)
        y_a1 = net(x, t, params, scale_factor=torch.ones(BATCH))
        y_a_highz = net(x, t, params, scale_factor=torch.full((BATCH,), 0.2))

    torch.testing.assert_close(y_none, y_a1, rtol=0, atol=0)
    # ... and the embedding is genuinely used, so a different a changes the output.
    assert not torch.allclose(y_none, y_a_highz, atol=1e-6)


def test_non_redshift_model_ignores_scale_factor():
    """Checkpoints trained without redshift conditioning must run unchanged even
    if a caller passes a scale factor."""
    net, in_ch, _ = _build(False, False, False, seed=3)
    x, t, params = _inputs(in_ch)

    with torch.no_grad():
        y_plain = net(x, t, params)
        y_with_a = net(x, t, params, scale_factor=torch.full((BATCH,), 0.2))

    assert net.redshift_emb is None
    torch.testing.assert_close(y_plain, y_with_a, rtol=0, atol=0)


def test_conditioning_actually_reaches_the_output():
    """params and t both drive AdaGroupNorm; a change in either must propagate."""
    net, in_ch, _ = _build(False, False, False, seed=4)
    x, t, params = _inputs(in_ch)

    with torch.no_grad():
        base = net(x, t, params)
        other_params = net(x, t, torch.zeros_like(params))
        other_t = net(x, torch.zeros_like(t), params)

    assert not torch.allclose(base, other_params, atol=1e-6)
    assert not torch.allclose(base, other_t, atol=1e-6)


def test_observable_conditioning_changes_only_the_param_encoder():
    """Observable conditioning reuses the same encoder with n_params = N_OBS."""
    from bind.data import N_OBS

    torch.manual_seed(5)
    net = UNet(in_ch=7, out_ch=3, **{**TINY, "n_params": N_OBS})
    x, t, _ = _inputs(7)
    obs = torch.rand(BATCH, N_OBS)
    with torch.no_grad():
        y = net(x, t, obs)
    assert y.shape == (BATCH, 3, RES, RES)
    assert net.param_emb.net[0].in_features == N_OBS


def test_flow_matching_sample_returns_target_shape():
    """End-to-end FlowMatching.sample on the tiny UNet: noise -> hydro state."""
    from bind.model import FlowMatching

    net, in_ch, out_ch = _build(True, False, False, seed=6)
    fm = FlowMatching(net, cfg_dropout=0.0, out_channels=out_ch, stars_two_head=True)
    g = torch.Generator().manual_seed(7)
    condition = torch.randn(BATCH, 1, RES, RES, generator=g)
    large_scale = torch.randn(BATCH, 3, RES, RES, generator=g)
    params = torch.rand(BATCH, N_PARAMS, generator=g)

    with torch.no_grad():
        out = fm.sample(condition, large_scale=large_scale, params=params, n_steps=2)

    assert out.shape == (BATCH, out_ch, RES, RES)
    assert torch.isfinite(out).all()
    assert np.isfinite(out.numpy()).all()
