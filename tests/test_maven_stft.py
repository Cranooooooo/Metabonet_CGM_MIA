"""The two properties MAVEN's whole construction rests on, checked numerically.

WHY THIS EXISTS
---------------
`vendor/MAVEN/visual_transforms.py` had its torchaudio Spectrogram/InverseSpectrogram
swapped for native `torch.stft`/`torch.istft`, because torchaudio is not in this
environment and installing it risks the torch build the rest of the pipeline is
validated against. The swap is only safe if the replacement is the SAME transform.

The trap is the window. `torchaudio.transforms.Spectrogram` defaults to a periodic
Hann window; `torch.stft` defaults to a RECTANGULAR one. A rectangular window at 50 %
overlap does not invert exactly, and the failure is quiet -- samples still decode, they
are just wrong at the frame boundaries. So the round trip is asserted, not assumed.

The second property is the one the flow-matching path needs. `encode` is affine, not
linear (`_norm` subtracts 1 after scaling), so `Phi(ax) = a Phi(x)` is FALSE. What is
true, and what is all the design uses, is that the constant cancels under a CONVEX
combination:

    (1-t)(Lx0 + c) + t(Lx1 + c) = L((1-t)x0 + t x1) + c

which is exactly the linear FM path z_t = (1-t)Phi(x0) + t Phi(x1), and likewise the
velocity target v* = Phi(x1) - Phi(x0) where c cancels in the difference. Both are
checked below at several t, because "the transform is linear" as usually stated is
wrong here and a reader who tests the wrong identity will think the code is broken.

    python -m pytest tests/test_maven_stft.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "MAVEN"))

from visual_transforms import DelayEmbed, PeriodFold, STFTEmbedder  # noqa: E402


def _windows(n=16, T=288, D=1, seed=0):
    """Smooth, CGM-scaled test signal -- not white noise.

    An exact-reconstruction claim is easy to pass on noise and hard on a signal with
    real spectral content near the band edges, which is where a wrong window shows up.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T)[None, :, None]
    x = (0.2 * np.sin(t + rng.uniform(0, 6.28, (n, 1, D)))
         + 0.05 * np.sin(7.3 * t) + 0.02 * rng.standard_normal((n, T, D)))
    return x.astype(np.float32)


@pytest.mark.parametrize("T,D,n_fft,hop", [(288, 1, 16, 8), (288, 2, 16, 8),
                                           (2016, 1, 96, 48), (2016, 2, 96, 48)])
def test_stft_round_trip_is_exact(T, D, n_fft, hop):
    """decode(encode(x)) == x to float32 round-off."""
    x = _windows(8, T, D)
    phi = STFTEmbedder(T, D, n_fft=n_fft, hop_length=hop)
    phi.fit(x)
    xt = torch.from_numpy(x)
    back = phi.decode(phi.encode(xt))
    assert back.shape == xt.shape
    err = (back - xt).abs().max().item()
    # 1e-5 is float32 STFT round-off on this amplitude; a rectangular window instead
    # of Hann lands around 1e-1 here, so this threshold separates the two cases by
    # four orders of magnitude rather than splitting hairs.
    assert err < 1e-5, f"round-trip error {err:.2e} -- window or centering is wrong"


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_convex_combination_passes_through_encode(t):
    """Phi((1-t)x0 + t x1) == (1-t)Phi(x0) + t Phi(x1): the FM path identity."""
    T, D = 288, 1
    x0, x1 = _windows(8, T, D, seed=1), _windows(8, T, D, seed=2)
    phi = STFTEmbedder(T, D, n_fft=16, hop_length=8)
    phi.fit(np.concatenate([x0, x1]))
    a, b = torch.from_numpy(x0), torch.from_numpy(x1)
    lhs = phi.encode((1 - t) * a + t * b)
    rhs = (1 - t) * phi.encode(a) + t * phi.encode(b)
    assert (lhs - rhs).abs().max().item() < 1e-4


def test_velocity_target_is_a_difference_so_the_offset_cancels():
    """v* = Phi(x1) - Phi(x0) equals the encode of the difference, up to the offset.

    Stated as the design uses it: the affine constant is absent from v*, so the
    velocity target carries no dependence on the per-bin normalisation offset.
    """
    T, D = 288, 1
    x0, x1 = _windows(8, T, D, seed=3), _windows(8, T, D, seed=4)
    phi = STFTEmbedder(T, D, n_fft=16, hop_length=8)
    phi.fit(np.concatenate([x0, x1]))
    a, b = torch.from_numpy(x0), torch.from_numpy(x1)
    v = phi.encode(b) - phi.encode(a)
    # encode(b) - encode(a) = L(b) - L(a) = L(b - a); adding the offset back gives
    # encode(b - a). Checking it this way tests the cancellation, not just linearity.
    zero = torch.zeros_like(a)
    assert (v - (phi.encode(b - a) - phi.encode(zero))).abs().max().item() < 1e-4


@pytest.mark.parametrize("T,P", [(2016, 288), (288, 288)])
def test_day_fold_has_no_padding_cells(T, P):
    """2016 = 7*288 folds exactly; a padded fold would exclude cells from every loss."""
    phi = PeriodFold(T, P)
    assert phi.L == T, f"L={phi.L} != T={T}: the fold padded, so cells are dropped"
    assert int(phi.flat_idx.min()) >= 0, "padding cells (flat_idx=-1) present"
    assert float(phi.valid.min()) == 1.0, "some tokens are padding"


@pytest.mark.parametrize("T,D", [(288, 1), (2016, 2)])
def test_gather_views_round_trip(T, D):
    """Both gather transforms are exact copies, so the round trip is bit-exact."""
    x = _windows(4, T, D)
    xt = torch.from_numpy(x)
    for phi in (DelayEmbed(T, tau=4, m=8), PeriodFold(T, 288)):
        back = phi.decode(phi.encode(xt))
        assert (back - xt).abs().max().item() == 0.0, f"{phi.name} is not exact"


def test_day_fold_halves_the_token_count_at_seven_days():
    """Why the day fold is used at T=2016: half the tokens.

    An earlier version of this docstring claimed the delay view "does not fit on an
    A100-40GB", from a hand calculation of a materialised L-by-L attention matrix:
    ~33 GB per layer at L=4024. The measurement (scripts/pbs/M2_maven_smoke.pbs, step
    4) refuted it -- peak memory is 4.0 GB at L=2016 and 7.9 GB at L=4024, a ratio of
    1.98 against the 4.0 a quadratic cost would give. torch's TransformerEncoderLayer
    dispatches to scaled-dot-product attention, which never materialises that matrix,
    so MEMORY is linear in L. Both views train comfortably; the day fold is a 2x
    saving, not an enabler. Left here as a warning: the arithmetic was not wrong, the
    tensor it described does not exist.
    """
    assert DelayEmbed(2016, tau=4, m=8).L == 4024
    assert PeriodFold(2016, 288).L == 2016
