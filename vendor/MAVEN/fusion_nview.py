"""N-view retrieval-guided editor fusion (V3, multi-resolution).

Generalizes the 2-view editor (fusion_editor.py) to N >= 2 visual views.
Given a base view (strongest) and N-1 "other" views (all in [0,1], paired by
shared-noise sampling), the editor edits the base toward the real manifold:

    fused = clip( base + alpha * smooth(V([base, {o_i - base}, mu-base, sigma, med-base]) Uᵀ)
                  * base(1-base), 0, 1 )

Per-cell input has (N+3)*D channels: base, the N-1 inter-view divergences, and
the TS2Vec-retrieved neighbour mean/std/median (residual on base). U is the
shared low-rank channel basis (preserves cross-correlation). Losses, retrieval,
bounded-skew, cap-decay alpha, TSTR-MAML: identical to the 2-view editor.

Usage:
    python fusion_nview.py --base A.npy --others B.npy,C.npy,D.npy \
        --real truth.npy --out fused.npy [--corr_weight 10] [--gpu 0]
"""
import argparse, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.func import functional_call

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Eval_Assembly'))
from fusion_editor import mmd2, corr_loss, _PFake, _pred_slice  # reuse


class FusionEditorN(nn.Module):
    """Rank-r channel-coupled residual editor over N views (base + N-1 others)."""
    def __init__(self, T, D, n_views, hidden=128, n_layers=2, n_heads=4, rank=8,
                 alpha_max=0.003, warmup_frac=0.15, alpha_end_frac=0.25, smooth_sigma=3.0):
        super().__init__()
        self.T, self.D, self.r, self.n_views = int(T), int(D), int(rank), int(n_views)
        in_blocks = n_views + 3          # base + (N-1) divergences + mean + std + med
        self.in_proj = nn.Linear(in_blocks * D, hidden)
        self.pos = nn.Parameter(torch.zeros(1, T, hidden)); nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=hidden*4,
                                           activation='gelu', batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(hidden)
        self.V_head = nn.Linear(hidden, self.r)
        self.U = nn.Parameter(torch.empty(D, self.r)); nn.init.trunc_normal_(self.U, std=1.0/(D**0.5))
        self.alpha_param = nn.Parameter(torch.tensor(0.0))
        self.alpha_max, self.warmup_frac, self.alpha_end_frac = float(alpha_max), float(warmup_frac), float(alpha_end_frac)
        self.register_buffer('_step', torch.zeros((), dtype=torch.long))
        self.register_buffer('_total_steps', torch.tensor(1, dtype=torch.long))
        ks = int(2*round(3*smooth_sigma)+1); t = torch.arange(ks)-ks//2
        k = torch.exp(-(t.float()**2)/(2*smooth_sigma**2)); k = (k/k.sum()).view(1,1,ks).repeat(self.r,1,1)
        self.register_buffer('smooth_k', k); self.smooth_pad = ks//2

    def set_schedule(self, n): self._total_steps.fill_(int(n))

    def _alpha(self):
        frac = (self._step.float()/self._total_steps.float()).clamp(0,1); wf, ef = self.warmup_frac, self.alpha_end_frac
        pi = torch.tensor(math.pi, device=frac.device)
        if frac.item() <= wf: sched = 0.5*(1-torch.cos(pi*(frac/wf).clamp(0,1)))
        else: sched = ef + (1-ef)*0.5*(1+torch.cos(pi*((frac-wf)/max(1e-6,1-wf)).clamp(0,1)))
        gain = F.softplus(self.alpha_param)/F.softplus(torch.zeros_like(self.alpha_param))
        return self.alpha_max*sched*gain

    def forward(self, base, others, n_mean, n_std, n_med):
        # others: list of (N-1) tensors [B,T,D]
        blocks = [base] + [o - base for o in others] + [n_mean - base, n_std, n_med - base]
        h = self.in_proj(torch.cat(blocks, dim=-1)) + self.pos
        h = self.norm_out(self.encoder(h))
        V = self.V_head(h)
        V_s = F.conv1d(V.transpose(1,2), self.smooth_k, padding=self.smooth_pad, groups=self.r).transpose(1,2)
        corr = V_s @ self.U.t()
        if self.training: self._step += 1
        alpha = self._alpha(); bound = base*(1.0-base)
        return (base + alpha*corr*bound).clamp(0,1), alpha


def build_retrieval(real, base_q, k, gpu, cache):
    if cache and os.path.exists(cache):
        return np.load(cache)
    from utils.ts2vec_loader import get_TS2Vec_class
    TS2Vec = get_TS2Vec_class()
    enc = TS2Vec(input_dims=real.shape[-1], device=int(gpu), batch_size=8, lr=1e-3, output_dims=320, max_train_length=3000)
    enc.fit(real, verbose=False)
    re = enc.encode(real, encoding_window='full_series'); qe = enc.encode(base_q, encoding_window='full_series')
    _, idx = NearestNeighbors(n_neighbors=k, metric='cosine').fit(re).kneighbors(qe)
    idx = idx.astype(np.int64)
    if cache: np.save(cache, idx)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='npy of base (strongest) view, [N,T,D] in [0,1]')
    ap.add_argument('--others', required=True, help='comma-separated npy paths of the other views')
    ap.add_argument('--real', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', default='0'); ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--k_neighbors', type=int, default=16); ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=128); ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--alpha_max', type=float, default=0.003)
    ap.add_argument('--mmd_weight', type=float, default=1.0); ap.add_argument('--corr_weight', type=float, default=10.0)
    ap.add_argument('--anchor_start', type=float, default=1.0); ap.add_argument('--anchor_end', type=float, default=0.05)
    ap.add_argument('--pred_weight', type=float, default=1.0); ap.add_argument('--pred_lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=2023)
    a = ap.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', a.gpu); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    base = np.load(a.base).astype('float32')
    others = [np.load(p).astype('float32') for p in a.others.split(',')]
    real = np.load(a.real).astype('float32')
    n = min([base.shape[0]] + [o.shape[0] for o in others]); N_real, T, D = real.shape
    base = base[:n]; others = [o[:n] for o in others]; n_views = 1 + len(others)
    print(f'[nview] base {base.shape} + {len(others)} others; N_views={n_views} T={T} D={D}', flush=True)

    base_t = torch.from_numpy(base).to(dev); others_t = [torch.from_numpy(o).to(dev) for o in others]; real_t = torch.from_numpy(real).to(dev)
    cache = a.base.replace('.npy', f'_ts2vec_knn_K{a.k_neighbors}.npy')
    nbr = torch.from_numpy(build_retrieval(real, base, a.k_neighbors, a.gpu, cache)).to(dev)
    print(f'[nview] retrieval {nbr.shape}', flush=True)

    ed = FusionEditorN(T, D, n_views, hidden=a.hidden, rank=a.rank, alpha_max=a.alpha_max).to(dev)
    steps = a.epochs * max(1, n // a.batch_size); ed.set_schedule(steps)
    opt = torch.optim.AdamW(ed.parameters(), lr=a.lr, weight_decay=1e-4)
    P = _PFake(D, max(1, D//2)).to(dev); popt = torch.optim.Adam(P.parameters(), lr=a.pred_lr)
    anneal = max(1, int(0.5*steps)); log_every = max(50, steps//20)
    ed.train(); P.train(); t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, n, (a.batch_size,), device=dev); rdx = torch.randint(0, N_real, (a.batch_size,), device=dev)
        xb = base_t[idx]; xo = [o[idx] for o in others_t]; xR = real_t[rdx]
        nb = real_t[nbr[idx]]; n_mean = nb.mean(1); n_std = nb.std(1)+1e-6; n_med = nb.median(1).values
        fused, alpha = ed(xb, xo, n_mean, n_std, n_med)
        Xf_d, Yf_d = _pred_slice(fused.detach(), D); popt.zero_grad(); F.l1_loss(P(Xf_d), Yf_d).backward(); popt.step()
        with torch.backends.cudnn.flags(enabled=False):
            Xf, Yf = _pred_slice(fused, D); params = dict(P.named_parameters())
            g = torch.autograd.grad(F.l1_loss(P(Xf), Yf), list(params.values()), create_graph=True, retain_graph=True)
            fast = {k: p - a.pred_lr*gg for (k, p), gg in zip(params.items(), g)}
            Xr, Yr = _pred_slice(xR, D); L_pred = F.l1_loss(functional_call(P, fast, (Xr,)), Yr)
        lam = a.anchor_start + (s/anneal)*(a.anchor_end-a.anchor_start) if s < anneal else a.anchor_end
        loss = a.mmd_weight*mmd2(fused, xR) + a.corr_weight*corr_loss(fused, xR) + lam*F.mse_loss(fused, xb.detach()) + a.pred_weight*L_pred
        opt.zero_grad(); loss.backward(); opt.step()
        if (s+1) % log_every == 0: print(f'[nview] step {s+1}/{steps} ({time.time()-t0:.0f}s) loss={loss.item():.5f} a={alpha.item():.5f}', flush=True)

    ed.eval(); out = np.empty_like(base); ib = 256
    with torch.no_grad():
        for i in range(0, n, ib):
            xb = base_t[i:i+ib]; xo = [o[i:i+ib] for o in others_t]; nb = real_t[nbr[i:i+ib]]
            f, _ = ed(xb, xo, nb.mean(1), nb.std(1)+1e-6, nb.median(1).values)
            out[i:i+ib] = f.cpu().numpy()
    np.save(a.out, out.astype('float32'))
    print(f'[nview] saved -> {a.out} {out.shape} range[{out.min():.3f},{out.max():.3f}]', flush=True)


if __name__ == '__main__':
    main()
