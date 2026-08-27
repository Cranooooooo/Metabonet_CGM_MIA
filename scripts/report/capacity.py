"""Is 'hidden_size 256 everywhere' the same capacity at T=288 and T=2016?

hidden_size is one knob. A sequence model's capacity also depends on what scales with T
-- positional embeddings, any length-dependent projection -- and on how much of that
capacity each timestep gets. Counting the checkpoint answers it.
"""
import torch, collections, re
from pathlib import Path
def load(c):
    p = sorted(Path(f'results/runs/matrix_{c}/base').glob('ckpt_*/checkpoint-10.pt'))
    if not p: return None
    sd = torch.load(str(p[0]), map_location='cpu')
    for k in ('model', 'state_dict', 'ema'):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]; break
    return {k: v for k, v in sd.items() if hasattr(v, 'shape')}
A, B = load('d1_c1'), load('d7_c1')
if A is None or B is None:
    raise SystemExit('checkpoint missing')
na = sum(v.numel() for v in A.values()); nb = sum(v.numel() for v in B.values())
print(f"  d1_c1 (T=288)   {len(A):4} 张量  {na:>12,} 参数")
print(f"  d7_c1 (T=2016)  {len(B):4} 张量  {nb:>12,} 参数   差 {nb-na:+,} ({100*(nb-na)/na:+.1f}%)")
print(f"\n  每个时间步分到的参数: d1 {na/288:,.0f}   d7 {nb/2016:,.0f}   比值 {(na/288)/(nb/2016):.1f}x")
print(f"\n  形状不同的张量(容量随 T 变的部分):")
diff = [(k, tuple(A[k].shape), tuple(B[k].shape)) for k in A if k in B and A[k].shape != B[k].shape]
for k, sa, sb in diff[:12]:
    print(f"    {k:52} {str(sa):20} -> {str(sb)}")
print(f"    共 {len(diff)} 个;它们贡献了 "
      f"{sum(B[k].numel()-A[k].numel() for k,_,_ in diff):+,} 参数")
only = set(A) ^ set(B)
if only: print(f"  只在一边出现的张量: {sorted(only)[:6]}")
