# 原版 IG-FM(不带任何隐私模块)在某个格子上的 27 个模型 —— 共用正文。
#
# 调用方必须设好:CELL / SHARDS / NS
#
# 为什么要这一批。目前 d1_c2 上只有【带模块】的 IG-FM,没有原版做对照,所以在那个
# 格子上分不出「IG-FM 本身不泄漏」和「是两个模块让它不泄漏」。d1_c1 上两者都测过,
# 结论是没有可测差异(而且原版本身就已经在地板上)—— 这一批是拿第二个格子去检验
# 那个结论,顺带给 IG-FM 家族在 d1_c2 上再添一个独立的 base(现在只有一个,
# 「每个 arm 只有一个 base、n_eff=1」是我们最大的方法学硬伤)。
#
# 参数逐字照抄 d1_c1 那批原版(results/runs/loo_igfm_d1_c1/base/meta.json):
# hidden 144 / steps 25000 / sampling_steps 500 / save_every 2000 / keep 2。
# 三个模块的开关一个都不给 —— 适配器默认全关,训练路径和原版逐位一致。
# micro_batch 显式写成 32(适配器默认值),这样它进 meta.json 也进参数指纹。
#
# base 在这里【要训】,不像带模块那两批是拷贝复用的:这个格子上原版的 base
# 还不存在。它是 job_order 的第 0 个,落在分片 0。
#
# ⚠️ env.sh 开了 set -euo pipefail;conda activate 可能把 -e 关掉,重开并验一遍。
set -euo pipefail
case $- in *e*) ;; *) echo "❌ errexit 没打开,拒绝继续"; exit 1;; esac
: "${CELL:?}"; : "${SHARDS:?}"; : "${NS:?}"

DSN=results/matrix/design/rep1
COH="data/cohort/matrix_$CELL"
OUT="results/runs/loo_igfm_$CELL"
ATK="results/matrix/attack/igfm_$CELL"
PARAMS='{"hidden":144,"steps":25000,"sampling_steps":500,"save_every":2000,"keep_checkpoints":2,"micro_batch":32}'

python - "$PARAMS" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
assert d["hidden"] == 144 and d["steps"] == 25000 and d["micro_batch"] == 32
for k in ("coord_mask", "level_dims", "order_dims", "iso_weight"):
    assert k not in d, f"原版不该出现 {k}"
print(f"[check] 原版参数正确,三个模块的键一个都没有: {d}")
PY

LOCK="$OUT/.analysis.lock"; DONE="$OUT/.analysis.done"; STAMP="$OUT/.params.json"
LOCK_HELD=0
mkdir -p logs "$OUT"

count_done() { local n=0 d; for d in "$OUT"/*/; do
  if [ -f "$d/samples.npy" ]; then n=$((n + 1)); fi; done; echo "$n"; }

on_exit() {
  if [ "$LOCK_HELD" = "1" ] && [ ! -f "$DONE" ]; then
    rmdir "$LOCK" 2>/dev/null || true; echo "== 分析未完成,已归还锁,重排队会重试"
  fi
  echo "== 收尾:$(count_done) / 27 个模型完成" || true
}
trap on_exit EXIT

# 参数指纹:run_loo 的跳过逻辑只比对 subjects_sha1(只来自 design,四个格子完全
# 相同),不看 params;适配器续训也不检查参数。改了参数再重排队,树里就一半旧配置
# 一半新配置,攻击照跑照出数。
if [ -f "$STAMP" ]; then
  [ "$(cat "$STAMP")" = "$PARAMS" ] || { echo "❌ 和树里已有模型的参数不一致"; \
    echo "   树里: $(cat "$STAMP")"; echo "   现在: $PARAMS"; exit 1; }
else printf '%s' "$PARAMS" > "$STAMP"; fi

IFS=',' read -ra ALLOC <<< "${CUDA_VISIBLE_DEVICES:-0}"
NG=${#ALLOC[@]}; read -ra SH <<< "$SHARDS"
[ "$NG" -eq "${#SH[@]}" ] || { echo "❌ 拿到 $NG 张卡但要跑 ${#SH[@]} 个分片,停"; exit 1; }
export THREADS=$(( ${NCPUS:-16} / NG ))
# ⚠️ 这一行其实是空转:env.sh 早就按 NCPUS 导出过 OMP/NUMBA/MKL_NUM_THREADS,
#    这里改 THREADS 不会重新导出它们(日志里能看到 threads=64)。和 d1_c1 那批、
#    以及带模块那批完全一样的空转,【刻意不改】—— 4.63 小时/模型这个墙钟预算
#    就是在这个 4 倍线程超订阅下实测出来的,把它「修好」等于作废掉那个基准。

echo "===== $(date '+%F %T')  $CELL · 原版 IG-FM(无模块)· 泄漏实验 ====="
echo "  分片 $SHARDS / $NS   卡 $NG 张   已完成 $(count_done) / 27"

pids=(); rc=0
for k in "${!SH[@]}"; do
  s="${SH[$k]}"
  ( export CUDA_VISIBLE_DEVICES="${ALLOC[$k]}"
    python -u scripts/run_loo.py --design "$DSN" --cohort "$COH" --out "$OUT" \
      --generator igfm --shard "$s" --n-shards "$NS" --seed 2026 --params "$PARAMS"
  ) > "logs/igfm_stock_loo_${CELL}_shard${s}.log" 2>&1 &
  pids+=($!)
done
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then echo "分片 ${SH[$k]} OK"; else echo "分片 ${SH[$k]} 失败"; rc=1; fi
done

n=$(count_done); echo "训练完成: $n / 27"

if [ "$n" -eq 27 ] && [ ! -f "$DONE" ] && mkdir "$LOCK" 2>/dev/null; then
  LOCK_HELD=1; ok=1
  A="logs/igfm_stock_attack_$CELL.log"; B="logs/igfm_stock_subjauc_$CELL.log"
  C="logs/igfm_stock_floor_$CELL.log"; Q="logs/igfm_stock_quality_$CELL.log"
  echo "== 攻击(和其它 arm 完全同一套冻结统计量)"
  # run_attack 在配对缺模型时不报错,只在输出最前面打一句警告然后照样写 summary,
  # 里面的 AUC 是子集上算的。所以全量进独立日志再检查那句话。
  python -u scripts/run_attack.py --design "$DSN" --cohort "$COH" --runs "$OUT" \
      --out "$ATK" --generator igfm > "$A" 2>&1 || ok=0
  if grep -q "missing a model" "$A"; then
    echo "❌ 有配对缺模型,summary 是子集,不可用:"; grep "missing a model" "$A"; ok=0; fi
  tail -30 "$A"
  echo "== 逐病人"
  python -u scripts/subject_auc.py --runs "$OUT" --design results/matrix/design \
      --cohort "$COH" --replicates 1 --out "results/matrix/subject_auc/igfm_$CELL" > "$B" 2>&1 || ok=0
  tail -20 "$B"
  echo "== 打乱地板(PITFALLS 第 20 条;地板是每个 release 各算各的)"
  python -u scripts/leak_locus.py --design "$DSN" --cohort "$COH" --runs "$OUT" \
      --out "results/matrix/leak_locus/igfm_$CELL.json" > "$C" 2>&1 || ok=0
  tail -20 "$C"
  # 这一批的 base 是新训的,顺手打个质量分,才好和同格子带模块的那个 base 比。
  echo "== base 质量分"
  mkdir -p "results/igfm_stock_$CELL"
  python -u scripts/eval_quality_tsgem.py --runs "$OUT/base" --cohort "$COH" \
      --design "$DSN" --out "results/igfm_stock_$CELL/tsgem.json" \
      --label "igfm_stock_$CELL" > "$Q" 2>&1 || ok=0
  grep -q context_fid "results/igfm_stock_$CELL/tsgem.json" 2>/dev/null || { echo "❌ 质量分是空的"; ok=0; }
  echo "  原版      $(grep -oE '"context_fid": *[0-9.]+' "results/igfm_stock_$CELL/tsgem.json" 2>/dev/null | head -1)"
  echo "  带两模块  $(grep -oE '"context_fid": *[0-9.]+' "results/igfm_priv_$CELL/tsgem.json" 2>/dev/null | head -1)"
  if [ "$ok" -eq 1 ]; then : > "$DONE"; echo "IGFM STOCK LOO $CELL 分析全部完成"
  else echo "❌ 分析失败(锁会在 trap 里归还)"; rc=1; fi
elif [ "$n" -eq 27 ] && [ -f "$DONE" ]; then
  echo "27 个齐了,分析此前已完整跑过。"
elif [ "$n" -eq 27 ]; then
  echo "27 个齐了,但分析锁被别的作业占着。"
  if [ ! -f "$ATK/summary.json" ]; then
    echo "❌ 而且 $ATK/summary.json 不存在 —— 可能已是死锁。确认无别的作业在跑后 rmdir $LOCK"; rc=1; fi
else
  echo "不足 27 个,不做分析。重排队会接着跑。"; rc=1
fi
[ "$rc" -eq 0 ] || echo "== 本作业退出码 $rc,上面有失败项,别当成功读"
exit $rc
