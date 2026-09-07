# 一个或多个 baseline 的 base 模型 —— 所有 baseline 作业共用的正文。
# 调用方设 JOBS="gen:cell gen:cell ..."(在同一张卡上依次跑)。
#
# 等预算口径:所有生成器都看【640 万条序列】,和 DiM-TS(10万步x64)、
# IG-FM(25000x256)对齐。PLAN.md 记着一次教训 —— Diffusion-TS 当初跑作者默认的
# 12000 步、DiM-TS 跑 100000 步,"差 20.7 倍"那个结论里预算和架构一起变了,不干净。
#
# ⚠️ 一个单位喂多少条,三个生成器不一样,搞反差 90 倍:
#     diffusion_ts  max_epochs  -> vendor 的 train_num_steps,是【迭代】,每单位 batch 条
#                                  (还要把 gradient_accumulate_every 设成 1,默认 2 会翻倍)
#     diffwave      total_iters -> 【迭代】,优先于 max_epochs(适配器 :121)
#     fourier_diff  max_epochs  -> pl.Trainer 的【真 epoch】,每单位 (N//batch)*batch 条
#
# ⚠️ 采样旋钮【不能】通过 --params 传给 fourier_diff / diffwave:它们的 sample() 是
#    `cfg = dict(sample_cfg or {})`,不合并构造参数,传了会被静默丢弃。run_loo 不传
#    sample_cfg,所以它们用各自的硬编码默认(fourier 1000 步/batch128,diffwave batch128)
#    —— 探针量的正是这一组,所以成本估计可以直接转移。别在这里"顺手"传。

set -euo pipefail
case $- in *e*) ;; *) echo "❌ errexit 没打开"; exit 1;; esac
: "${JOBS:?}"

DSN=results/matrix/design/rep1
BUDGET=6400000

params_for() {          # $1=gen $2=cell
  local g="$1" c="$2" bs=64
  case "$c" in d7_*) bs=8 ;; esac      # T=2016 上 batch 64 必 OOM(探针实测)
  case "$g" in
    diffusion_ts) echo "{\"max_epochs\":$((BUDGET/bs)),\"batch_size\":$bs,\"gradient_accumulate_every\":1,\"timesteps\":500}" ;;
    diffwave)     echo "{\"total_iters\":$((BUDGET/bs)),\"batch_size\":$bs}" ;;
    fourier_diff) local n=5741; local per=$n   # vendor 的 DataLoader 没有 drop_last(datamodules.py),一个 epoch 就是 N 条
                  echo "{\"max_epochs\":$(( (BUDGET+per-1)/per )),\"batch_size\":$bs}" ;;
    *) echo "❌ 不认识的生成器 $g" >&2; return 1 ;;
  esac
}

rc=0
for j in $JOBS; do
  g="${j%%:*}"; c="${j##*:}"
  OUT="results/runs/bl_${g}_${c}"; QOUT="results/bl_${g}_${c}"
  COH="data/cohort/matrix_$c"
  P=$(params_for "$g" "$c") || { rc=1; continue; }
  mkdir -p "$OUT" "$QOUT"
  echo; echo "===== $(date '+%F %T')  $g / $c ====="
  echo "参数: $P"
  python - "$P" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
print(f"[check] params 解析正确: {d}")
PY
  # 参数指纹:run_loo 的跳过逻辑只比对 subjects_sha1(只来自 design,四个格子相同),
  # 不看 params,也不看 cohort。改了预算再重排队会一半旧一半新。
  ST="$OUT/.params.json"
  if [ -f "$ST" ]; then
    [ "$(cat "$ST")" = "$P" ] || { echo "❌ 参数和树里已有的不一致"; echo "  已有: $(cat "$ST")"; echo "  现在: $P"; rc=1; continue; }
  else printf '%s' "$P" > "$ST"; fi

  if python -u scripts/run_loo.py --design "$DSN" --cohort "$COH" --out "$OUT" \
       --generator "$g" --job base --seed 2026 --params "$P"; then
    if [ -s "$OUT/base/samples.npy" ]; then
      python -u scripts/eval_quality_tsgem.py --runs "$OUT/base" --cohort "$COH" \
          --design "$DSN" --out "$QOUT/tsgem.json" --label "bl_${g}_${c}" || rc=1
      # gate 必须看【数字】不是【键】:eval_quality_tsgem 会写出 "context_fid": null,
      # grep -q 那个键照样成功,于是一个失败的指标被当成通过报出去。
      CF=$(grep -oE '"context_fid": *[0-9.eE+-]+' "$QOUT/tsgem.json" 2>/dev/null | head -1 || true)
      if [ -n "$CF" ]; then echo "  >>> $g/$c  $CF"
      else echo "❌ $g/$c 打分没有可读的 context_fid(可能是 null)"; rc=1; fi
      # 参照(同格子):IG-FM+模块 和 DiM-TS。跨格子不要比绝对值。
      REF=$(grep -oE '"context_fid": *[0-9.eE+-]+' "results/igfm_priv_$c/tsgem.json" 2>/dev/null | head -1 || true)
      echo "  参照 IG-FM+模块 ${REF:-n/a}"   # 空输入时 head 退 0,所以 || echo 永远走不到
    else echo "❌ $g/$c 没产出样本"; rc=1; fi
  else echo "❌ $g/$c 训练失败"; rc=1; fi
done
echo; echo "== 收尾 $(date '+%F %T'),退出码 $rc"
exit $rc
