#!/usr/bin/env python
"""Diffusion-TS 在几档容量下的参数量 —— 纯构建,不训练,CPU 上几秒。

为什么需要:我们的适配器默认是 n_layer_enc=1, n_layer_dec=1, d_model=32(223,719 参数),
而作者自己发布的配置里【最小的 d_model 是 64、n_layer_dec 最少是 2】——
vendor/Diffusion-TS/Config/ 下 11 个 yaml 没有一个用 32。所以此前所有
"Diffusion-TS 质量差" 的结论都是在一个作者从未用过、且小于其所有配置的容量上得到的。
参照:IG-FM 2.09M, DiM-TS 2.76M, DiffWave 2.72M。
"""
import sys
from cgmoutlier._env import check as _envcheck
_envcheck()
from cgmoutlier.generators.registry import get as get_generator

GRID = [
    ("适配器默认(此前所有结论)",        dict(n_layer_enc=1, n_layer_dec=1, d_model=32)),
    ("作者 config.yaml 模板(len160)",   dict(n_layer_enc=1, n_layer_dec=2, d_model=64)),
    ("作者 stocks",                     dict(n_layer_enc=2, n_layer_dec=2, d_model=64)),
    ("作者 mujoco_sssd(len100)",        dict(n_layer_enc=3, n_layer_dec=3, d_model=64)),
    ("作者 energy",                     dict(n_layer_enc=4, n_layer_dec=3, d_model=96)),
    ("作者 solar(len192,最长序列)",     dict(n_layer_enc=4, n_layer_dec=4, d_model=96)),
    ("中间档 A",                        dict(n_layer_enc=2, n_layer_dec=2, d_model=96)),
    ("中间档 B",                        dict(n_layer_enc=3, n_layer_dec=2, d_model=96)),
    ("中间档 C",                        dict(n_layer_enc=2, n_layer_dec=3, d_model=128)),
]
T, C = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (288, 1)
print(f"T={T} C={C}   参照: IG-FM 2,090,000 / DiM-TS 2,762,291 / DiffWave 2,718,977\n")
print(f"{'配置':<30}{'enc/dec/d_model':<20}{'参数量':>12}{'vs IG-FM':>10}")
for name, p in GRID:
    p = dict(p, n_heads=4, mlp_hidden_times=4, timesteps=500)
    g = get_generator("diffusion_ts")(T=T, C=C, params=p, device="cpu", seed=2026)
    # _build_model 用的是 self._device,而那个属性只有 fit() 才会设 —— 这里只构建不训练,手动补上。
    import torch; g._device = torch.device("cpu")
    n = sum(q.numel() for q in g._build_model().parameters())
    tag = f"{p['n_layer_enc']}/{p['n_layer_dec']}/{p['d_model']}"
    print(f"{name:<30}{tag:<20}{n:>12,}{n/2_090_000:>9.2f}x")
