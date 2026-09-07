#!/bin/bash
# 三个 baseline 的 26 个 include 模型。GEN / CELL / NSHARD 由外层 PBS 给。
#
# 为什么可以反复重排同一个作业:run_loo 以【每个模型的 samples.npy 是否存在】
# 决定跳过,所以 2 小时被切掉只丢当前那一个模型,已完成的保留。这三个适配器
# 【没有断点续训】,所以模型内部没有部分学分 —— 但模型之间有。
#
# base 已经训好在同一个 OUT 目录里(bl_<gen>_<cell>/base),等预算参数必须逐字
# 相同,否则 base 和 include 训到不同长度,泄漏差就混进了训练时长 —— 那正是
# 整个设计要排除的唯一混淆。
set -uo pipefail
: "${GEN:?}"; : "${CELL:?}"; : "${NSHARD:=4}"
cd "${REPO:-$HOME/workspace/Project_CGM/CGM-OutlierMIA-master}"
export PYTHONNOUSERSITE=1

BUDGET=6400000; BS=64
case "$GEN" in
  diffusion_ts) P="{\"max_epochs\":$((BUDGET/BS)),\"batch_size\":$BS,\"gradient_accumulate_every\":1,\"timesteps\":500}" ;;
  diffwave)     P="{\"total_iters\":$((BUDGET/BS)),\"batch_size\":$BS}" ;;
  fourier_diff) P="{\"max_epochs\":$(( (BUDGET+5741-1)/5741 )),\"batch_size\":$BS}" ;;
  *) echo "❌ 不认识的生成器 $GEN"; exit 2 ;;
esac
OUT="results/runs/bl_${GEN}_${CELL}"; COH="data/cohort/matrix_$CELL"
DSN=results/matrix/design/rep1

# base 必须已经存在且参数一致 —— 不检查的话会拿一个别的预算训出来的 base
# 当对照,而 meta.json 看上去完全正常。
if [ ! -s "$OUT/base/samples.npy" ]; then echo "❌ $OUT/base 还没有样本,先训 base"; exit 3; fi
python - "$OUT/base/meta.json" "$P" <<'PY' || exit 3
import json, sys
have = json.load(open(sys.argv[1]))["params"]; want = json.loads(sys.argv[2])
bad = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
print(f"[check] base 参数比对: {'一致' if not bad else bad}")
sys.exit(1 if bad else 0)
PY

echo "===== $(date '+%F %T')  $GEN / $CELL  ${NSHARD} 分片 ====="
echo "参数: $P"
python -u scripts/run_loo.py --design "$DSN" --cohort "$COH" --out "$OUT" \
    --generator "$GEN" --params "$P" --seed 2026 --list | head -3

pids=()
for s in $(seq 0 $((NSHARD-1))); do
  CUDA_VISIBLE_DEVICES=$s python -u scripts/run_loo.py \
      --design "$DSN" --cohort "$COH" --out "$OUT" \
      --generator "$GEN" --params "$P" --seed 2026 \
      --shard "$s" --n-shards "$NSHARD" \
      > "logs/blloo_${GEN}_${CELL}_s${s}.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done

DONE=$(ls -d $OUT/*/ 2>/dev/null | while read d; do [ -s "$d/samples.npy" ] && echo x; done | wc -l)
echo "===== $(date '+%F %T')  完成 $DONE / 27  (退出码 $rc)"
echo "     未满 27 就再排一次这个作业,已完成的会被跳过"
