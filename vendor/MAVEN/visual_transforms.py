#!/usr/bin/env python3
"""visual_transforms.py — dual visual embeddings for MAVEN.

Implements the two image embeddings of New_Model_design.md §3:

  branch A : delay embedding   (ImagenTime style)  -> grid (m, W)
  branch B : period folding    (ts2img style)      -> grid (n_rows, P)

Both are *linear gather maps*: every image cell is copied from exactly one
time-series cell, so for the linear-path FM interpolation

    Phi(x_t) = (1-t) Phi(x0) + t Phi(x1)

holds exactly (design §3.1) and the velocity target / loss can be defined
directly in image space.  The inverse Phi^-1 is overlap-add: a TS cell that
several image cells map to is recovered by averaging them.

A transform is fully described by `flat_idx [L]` — for each of the L image
tokens, the TS index it gathers from (-1 marks a padding cell that has no TS
pre-image, e.g. the zero-padded tail of branch B; design risk #5).  Masks are
mapped to image space with the *same* gather (design §3.3), so the per-cell
FM loss support is exact with no approximation.

The Model_6 source (`dimts_data.py`, `Utils/ts2img.py`) is not present in this
repo; these transforms are re-implemented from the design spec and honour its
mathematical contract (linear gather + overlap-add inverse) and the feasibility
bound `T1_max = floor(1.5 * sqrt(T))` for the period of branch B.
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Base: a precomputed linear gather map between TS and image
# ============================================================
class GatherTransform(nn.Module):
    """Linear gather map TS [B,T,D] <-> image-token sequence [B,L,D].

    Stores everything as buffers so `.to(device)` moves the index tables.

    Attributes
    ----------
    flat_idx : LongTensor [L]   TS index each image token gathers (-1 = pad)
    valid    : FloatTensor [L]  1.0 for real tokens, 0.0 for padding tokens
    counts   : FloatTensor [T]  how many image tokens map onto each TS cell
    L        : int              number of image tokens (sequence length)
    T        : int              time-series window length
    grid     : (int, int)       2D image shape (H, W) before flattening
    """

    def __init__(self, flat_idx, T, grid_hw, name='gather'):
        super().__init__()
        flat_idx = torch.as_tensor(flat_idx, dtype=torch.long)
        valid = (flat_idx >= 0).float()
        counts = torch.zeros(T)
        real = flat_idx[flat_idx >= 0]
        counts.index_add_(0, real, torch.ones(real.numel()))
        if (counts == 0).any():
            missing = int((counts == 0).sum())
            raise ValueError(
                f'{name}: {missing} TS cell(s) are not covered by any image '
                f'cell — transform is not invertible. Check (tau,m)/(P).'
            )
        self.L = int(flat_idx.numel())
        self.T = int(T)
        self.grid = tuple(grid_hw)
        self.name = name
        # A gather map is cell-aligned: every image token <-> exactly one TS
        # cell, so a TS mask transfers exactly and the per-cell FM loss support
        # is exact. The STFT branch sets this False (no cell correspondence).
        self.cell_aligned = True
        self.register_buffer('flat_idx', flat_idx)
        self.register_buffer('valid', valid)
        self.register_buffer('counts', counts)

    # --- Phi : TS -> image -------------------------------------------------
    def encode(self, x):
        """x [B,T,D] -> image tokens [B,L,D]. Padding tokens are set to 0.

        Works for data tensors and for {0,1} masks alike (design §3.3)."""
        idx = self.flat_idx.clamp(min=0)               # [L]
        img = x[:, idx, :]                             # [B,L,D] gather
        return img * self.valid.view(1, -1, 1)

    # --- Phi^-1 : image -> TS (overlap-add) --------------------------------
    def decode(self, img):
        """image tokens [B,L,D] -> TS [B,T,D]. Redundant cells are averaged;
        padding tokens are dropped."""
        B, L, D = img.shape
        src = img * self.valid.view(1, -1, 1)
        idx = self.flat_idx.clamp(min=0).view(1, -1, 1).expand(B, L, D)
        out = torch.zeros(B, self.T, D, device=img.device, dtype=img.dtype)
        out.scatter_add_(1, idx, src)
        return out / self.counts.view(1, -1, 1)

    def image_valid(self, B, D, device):
        """M_obs in image space for full (non-missing) data: 1 on real
        tokens, 0 on padding tokens."""
        return self.valid.view(1, -1, 1).expand(B, self.L, D).to(device)

    def extra_repr(self):
        return (f'{self.name}: T={self.T} -> grid={self.grid} L={self.L} '
                f'(pad={int((self.valid == 0).sum())})')


# ============================================================
# Branch A: delay embedding  (phase-space / local trajectory)
# ============================================================
def make_delay_embed(T, tau, m):
    """Delay embedding grid (m rows, W cols): column j is the length-m window
    starting at `start_j`, i.e. cell (i,j) -> TS index start_j + i.

    Column starts step by `tau` and the last column is clamped to T-m so the
    whole window is covered. `tau <= m` guarantees full coverage. No padding."""
    if tau > m:
        raise ValueError(f'delay embed needs tau<=m for full coverage, '
                          f'got tau={tau} m={m}')
    W = math.ceil((T - m) / tau) + 1
    starts = [min(j * tau, T - m) for j in range(W)]
    flat = [starts[j] + i for i in range(m) for j in range(W)]   # row-major
    return flat, (m, W)


class DelayEmbed(GatherTransform):
    def __init__(self, T, tau=4, m=8):
        flat, grid = make_delay_embed(T, tau, m)
        super().__init__(flat, T, grid, name=f'delay(tau={tau},m={m})')
        self.tau, self.m = tau, m


# ============================================================
# Branch B: period folding  (inter-period / intra-period phase)
# ============================================================
def pick_period(windows, cap):
    """Dominant period of `windows` [N,T,D] from the mean rFFT power spectrum
    (DC bin excluded), clamped to [2, cap].  `cap = floor(1.5*sqrt(T))` is the
    design's feasibility bound on the fold period."""
    x = windows - windows.mean(axis=1, keepdims=True)
    power = (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(axis=(0, 2))
    power[0] = 0.0
    if power.shape[0] <= 1:
        return 2
    k = int(np.argmax(power[1:])) + 1
    T = windows.shape[1]
    return int(max(2, min(cap, round(T / k))))


def make_period_fold(T, P):
    """Period-fold grid (n_rows rows, P cols): cell (r,c) -> TS index r*P + c.
    The last row is zero-padded when T is not a multiple of P; those cells get
    flat_idx = -1 and are excluded from every loss (design risk #5)."""
    n_rows = math.ceil(T / P)
    flat = []
    for r in range(n_rows):
        for c in range(P):
            idx = r * P + c
            flat.append(idx if idx < T else -1)
    return flat, (n_rows, P)


class PeriodFold(GatherTransform):
    def __init__(self, T, P):
        flat, grid = make_period_fold(T, P)
        super().__init__(flat, T, grid, name=f'periodfold(P={P})')
        self.P = P


def t1_max(T):
    """Design feasibility bound for the branch-B fold period."""
    return int(math.floor(1.5 * math.sqrt(T)))


# ============================================================
# Branch B (alt): complex STFT  (ImagenTime style, arXiv 2410.19538)
# ============================================================
class STFTEmbedder(nn.Module):
    """Short-time Fourier transform TS <-> image, the ImagenTime recipe.

    Unlike the gather branches the STFT is a *dense* linear map (overlapping
    windowed DFT), not a per-cell copy — but it is still **linear**, so the
    linear-path FM identity `Phi(x_t) = (1-t)Phi(x0) + t Phi(x1)` holds, and a
    Hann window at 50% overlap (hop = n_fft/2) makes the inverse STFT an exact
    reconstruction. Magnitude-only spectra are *not* invertible (phase lost);
    we keep the full complex spectrum as real+imag, the fix the earlier
    magnitude-STFT attempt missed (design §1.3 / §12).

    Image layout: STFT gives `[B,D,F,frames]` complex. Real and imag parts are
    each MinMax-normalised per frequency bin to [-1,1] (an affine map — still
    linear-path safe) and concatenated along the frame axis into a real
    `[B,D,F,2*frames]` image, so the per-token feature dim stays `D` (same as
    the gather branches — the VelocityNet needs no change, only seq_len). Token
    sequence: `[B, F*2*frames, D]`.

    Not cell-aligned: a TS cell smears across frames, so TS masks do not
    transfer exactly — the cascade conditions via TS-domain quantities instead
    and the impute FM loss is taken over all cells (see train_maven.py).
    """

    def __init__(self, T, D, n_fft=16, hop_length=8):
        super().__init__()
        # LOCAL CHANGE (CGM-OutlierMIA): torchaudio's Spectrogram/InverseSpectrogram
        # replaced by native torch.stft/torch.istft. torchaudio is not in this repo's
        # environment, and installing it risks pulling a torch that does not match the
        # one the rest of the pipeline is validated against. The two are the same
        # transform: torchaudio's Spectrogram(power=None) IS torch.stft with a periodic
        # Hann window, center=True, pad_mode='reflect', onesided, un-normalised, and
        # InverseSpectrogram is torch.istft with the same. The window has to be passed
        # explicitly -- torch.stft defaults to a RECTANGULAR window, which would break
        # the exact round trip the linear-path FM identity depends on.
        # Verified to float32 round-off by tests/test_maven_stft.py.
        self.T, self.D = int(T), int(D)
        self.n_fft, self.hop = int(n_fft), int(hop_length)
        self.register_buffer('window', torch.hann_window(int(n_fft)))
        with torch.no_grad():
            probe = self.spec(torch.zeros(1, D, T))          # [1,D,F,frames]
        self.F, self.frames = int(probe.shape[-2]), int(probe.shape[-1])
        self.L = self.F * 2 * self.frames
        self.grid = (self.F, 2 * self.frames)
        self.cell_aligned = False
        self.name = f'stft(n_fft={n_fft},hop={hop_length})'
        # per-frequency-bin MinMax params (set by fit()); affine -> FM-safe.
        for k in ('min_real', 'max_real', 'min_imag', 'max_imag'):
            self.register_buffer(k, torch.zeros(self.F) if 'min' in k
                                 else torch.ones(self.F))

    def spec(self, x):
        """[N,D,T] real -> [N,D,F,frames] complex. Stands in for torchaudio."""
        N, D, T = x.shape
        z = torch.stft(x.reshape(N * D, T), n_fft=self.n_fft, hop_length=self.hop,
                       win_length=self.n_fft, window=self.window.to(x.device),
                       center=True, pad_mode='reflect', normalized=False,
                       onesided=True, return_complex=True)
        return z.reshape(N, D, z.shape[-2], z.shape[-1])

    def ispec(self, z, length):
        """[N,D,F,frames] complex -> [N,D,length] real."""
        N, D, F, fr = z.shape
        x = torch.istft(z.reshape(N * D, F, fr), n_fft=self.n_fft, hop_length=self.hop,
                        win_length=self.n_fft, window=self.window.to(z.device),
                        center=True, normalized=False, onesided=True, length=length)
        return x.reshape(N, D, length)

    @torch.no_grad()
    def fit(self, windows):
        """Cache per-frequency min/max of real & imag over the training set."""
        x = torch.as_tensor(np.asarray(windows), dtype=torch.float32)
        spec = self.spec(x.transpose(1, 2))                  # [N,D,F,frames]
        re, im = spec.real, spec.imag
        self.min_real.copy_(re.amin(dim=(0, 1, 3)))
        self.max_real.copy_(re.amax(dim=(0, 1, 3)))
        self.min_imag.copy_(im.amin(dim=(0, 1, 3)))
        self.max_imag.copy_(im.amax(dim=(0, 1, 3)))

    def _norm(self, v, mn, mx):                              # affine -> [-1,1]
        rng = (mx - mn).clamp(min=1e-6).view(1, 1, -1, 1)
        return (v - mn.view(1, 1, -1, 1)) / rng * 2.0 - 1.0

    def _denorm(self, v, mn, mx):
        rng = (mx - mn).clamp(min=1e-6).view(1, 1, -1, 1)
        return (v + 1.0) / 2.0 * rng + mn.view(1, 1, -1, 1)

    def encode(self, x):
        """TS [B,T,D] -> image tokens [B, F*2frames, D]."""
        spec = self.spec(x.transpose(1, 2))                  # [B,D,F,frames]
        re = self._norm(spec.real, self.min_real, self.max_real)
        im = self._norm(spec.imag, self.min_imag, self.max_imag)
        img = torch.cat([re, im], dim=-1)                    # [B,D,F,2frames]
        B, D, F, W = img.shape
        return img.reshape(B, D, F * W).transpose(1, 2)      # [B,L,D]

    def decode(self, tok):
        """image tokens [B,L,D] -> TS [B,T,D] (inverse STFT)."""
        B, L, D = tok.shape
        img = tok.transpose(1, 2).reshape(B, D, self.F, 2 * self.frames)
        re = self._denorm(img[..., :self.frames], self.min_real, self.max_real)
        im = self._denorm(img[..., self.frames:], self.min_imag, self.max_imag)
        xp = self.ispec(torch.complex(re, im), length=self.T)  # [B,D,T]
        return xp.transpose(1, 2)

    def image_valid(self, B, D, device):
        """STFT has no padding tokens — every cell is real."""
        return torch.ones(B, self.L, D, device=device)

    def extra_repr(self):
        return (f'{self.name}: T={self.T} -> grid={self.grid} L={self.L} '
                f'(F={self.F}, frames={self.frames})')


def build_transforms(windows, T, tau=4, m=8, period=None, branch_b='stft',
                     n_fft=16, hop_length=8):
    """Construct both visual branches for a dataset.

    windows  : [N,T,D] training windows (branch-B period / STFT min-max stats).
    branch_b : 'stft' (complex STFT, ImagenTime style — default) or
               'periodfold' (the original gather-based period fold).
    Returns (phi_A: DelayEmbed, phi_B: STFTEmbedder | PeriodFold).
    """
    phi_A = DelayEmbed(T, tau=tau, m=m)
    if branch_b == 'stft':
        phi_B = STFTEmbedder(T, windows.shape[-1], n_fft=n_fft, hop_length=hop_length)
        phi_B.fit(windows)
    elif branch_b == 'periodfold':
        if period is None:
            period = pick_period(windows, cap=t1_max(T))
        phi_B = PeriodFold(T, period)
    else:
        raise ValueError(f'unknown branch_b={branch_b!r}')
    return phi_A, phi_B
