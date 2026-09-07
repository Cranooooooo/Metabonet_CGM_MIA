# IG-FM + 两个隐私模块,某一个格子的 26 个 include 模型 —— 所有作业共用的正文。
#
# 调用方必须设好(都是硬性的,漏了立刻停):
#   CELL    d1_c2 / d7_c1 / d7_c2
#   EXTRA   额外并进 params 的 JSON 片段,必须含 micro_batch
#   NS      分片总数
#   SHARDS  本作业负责哪几片(空格分隔)
#
# 这是 igfm_priv_loo.body.sh(d1_c1 专用,已跑完,留作记录)的通用版,并补上了那次
# review 指出的下游隐患:原版的两道防漂移守卫【都不覆盖 micro_batch】。七天窗口的
# micro_batch 是探针量出来的 8,而适配器写死的默认值是 32 —— 照抄原版的话,base 用 8
# 训、26 个 include 用 32 训,而参数指纹和九个键的检查【全部通过】。这里把
# micro_batch 加进了指纹,也加进了对 base/meta.json 的逐键核对。
#
# ⚠️ env.sh 开了 set -euo pipefail,本文件是被它之后 source 的;conda activate 可能
#    把 -e 关掉,所以重新打开并验一遍。
set -euo pipefail
case $- in *e*) ;; *) echo "❌ errexit 没打开,拒绝继续"; exit 1;; esac
: "${CELL:?}"; : "${EXTRA:?}"; : "${NS:?}"; : "${SHARDS:?}"

DSN=results/matrix/design/rep1
COH="data/cohort/matrix_$CELL"
OUT="results/runs/loo_igfm_priv_$CELL"
ATK="results/matrix/attack/igfm_priv_$CELL"

BASEP=$(cat scripts/pbs/dev/igfm_priv.params | tr -d '\n' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
PARAMS="${BASEP%\}}, $EXTRA}"
python - "$PARAMS" <<'PY'
import json, sys
d = json.loads(sys.argv[1])          # 拼坏了在这里就崩,而不是四个分片各崩各的
assert "micro_batch" in d, "params 缺 micro_batch"
print(f"[check] params 解析正确: micro_batch={d['micro_batch']} "
      f"steps={d['steps']} keep_checkpoints={d['keep_checkpoints']}")
PY

LOCK="$OUT/.analysis.lock"; DONE="$OUT/.analysis.done"; STAMP="$OUT/.params.json"
LOCK_HELD=0
mkdir -p logs "$OUT"

# base 是从 igfm_priv_<cell>/base 拷进来的实体(不是符号链接:共享目录的话,任何一次
# 往那边的重跑都会悄悄换掉 26 个配对的对照,而 subjects_sha1 检查照样通过)。
# 点号开头的 .analysis.* / .params.json 不被 */ 匹配,不污染计数。
count_done() { local n=0 d; for d in "$OUT"/*/; do
  if [ -f "$d/samples.npy" ]; then n=$((n + 1)); fi; done; echo "$n"; }

# 唯一的 EXIT 处理:归还没用完的分析锁 + 报进度。锁只在「干净失败」时撤销的话,
# 墙钟 / 节点撤离 / qdel 会把它永久留在盘上,之后每次重排队都会看到 27 个齐、
# mkdir 失败、打印「已被接手」、然后 exit 0 —— 次次报绿,分析永远不跑。
on_exit() {
  if [ "$LOCK_HELD" = "1" ] && [ ! -f "$DONE" ]; then
    rmdir "$LOCK" 2>/dev/null || true; echo "== 分析未完成,已归还锁,重排队会重试"
  fi
  echo "== 收尾:$(count_done) / 27 个模型完成" || true
}
trap on_exit EXIT

# ---- 守卫一:参数漂移 ----
# run_loo 的跳过逻辑只比对 subjects_sha1(只来自 design,四个格子完全相同),
# 根本不看 params。改了参数再重排队,树里就一半旧配置一半新配置,攻击照跑照出数。
if [ -f "$STAMP" ]; then
  if [ "$(cat "$STAMP")" != "$PARAMS" ]; then
    echo "❌ 和这棵树里已有模型的参数不一致,停。"
    echo "   树里: $(cat "$STAMP")"; echo "   现在: $PARAMS"; exit 1
  fi
else
  printf '%s' "$PARAMS" > "$STAMP"
fi

# ---- 守卫二:base 必须是本格子、带对模块、【同一个 micro_batch】的那个 base ----
WANT=$(grep -oE '"subjects_sha1": *"[0-9a-f]+"' "$OUT/base/meta.json" 2>/dev/null \
       | grep -oE '[0-9a-f]{16}' || true)
[ "$WANT" = "8cf77b7732b8fe79" ] || { echo "❌ base 不是本设计的 base(读到 ${WANT:-空}),停"; exit 1; }
grep -q "\"cohort\": \"$COH\"" "$OUT/base/meta.json" \
  || { echo "❌ base 不是本格子($COH)的,停"; exit 1; }
MB=$(printf '%s' "$EXTRA" | grep -oE '"micro_batch" *: *[0-9]+' | grep -oE '[0-9]+$')
for kv in '"hidden": 144,' '"level_dims": 48,' '"order_dims": 48,' \
          '"steps": 25000,' '"sampling_steps": 500,' \
          '"coord_mask": true,' '"p_block": 0.35,' '"p_level": 0.25,' \
          '"iso_weight": false,' "\"micro_batch\": $MB,"; do
  grep -q "$kv" "$OUT/base/meta.json" || { echo "❌ base 的 params 缺 [$kv],停"; exit 1; }
done
echo "base 的 10 个关键参数全部对上(含 micro_batch=$MB)"

IFS=',' read -ra ALLOC <<< "${CUDA_VISIBLE_DEVICES:-0}"
NG=${#ALLOC[@]}; read -ra SH <<< "$SHARDS"
[ "$NG" -eq "${#SH[@]}" ] || { echo "❌ 拿到 $NG 张卡但要跑 ${#SH[@]} 个分片,停"; exit 1; }
export THREADS=$(( ${NCPUS:-16} / NG ))     # 和 d1_c1 那批一样的空转,刻意不改

echo "===== $(date '+%F %T')  $CELL · IG-FM + 改动一 + 改动二 · 泄漏实验 ====="
echo "  分片 $SHARDS / $NS   卡 $NG 张   已完成 $(count_done) / 27"
echo "  参数 $PARAMS"

pids=(); rc=0
for k in "${!SH[@]}"; do
  s="${SH[$k]}"
  ( export CUDA_VISIBLE_DEVICES="${ALLOC[$k]}"
    python -u scripts/run_loo.py --design "$DSN" --cohort "$COH" --out "$OUT" \
      --generator igfm --shard "$s" --n-shards "$NS" --seed 2026 --params "$PARAMS"
  ) > "logs/igfm_priv_loo_${CELL}_shard${s}.log" 2>&1 &
  pids+=($!)
done
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then echo "分片 ${SH[$k]} OK"; else echo "分片 ${SH[$k]} 失败"; rc=1; fi
done

n=$(count_done); echo "训练完成: $n / 27"

# 收尾靠「数够 27」而不是「我跑完了」—— 三个作业可能差几小时结束。mkdir 是原子的。
if [ "$n" -eq 27 ] && [ ! -f "$DONE" ] && mkdir "$LOCK" 2>/dev/null; then
  LOCK_HELD=1; ok=1
  A="logs/igfm_priv_attack_$CELL.log"; B="logs/igfm_priv_subjauc_$CELL.log"; F="logs/igfm_priv_floor_$CELL.log"
  echo "== 攻击(和原版基线完全同一套冻结统计量)"
  # run_attack 在某个配对缺模型时【不报错】:打一句警告到最前面、跳过、照样写出
  # summary.json,里面的 AUC 是子集上算的。tail 一定看不到那句。所以全量进独立日志。
  python -u scripts/run_attack.py --design "$DSN" --cohort "$COH" --runs "$OUT" \
      --out "$ATK" --generator igfm > "$A" 2>&1 || ok=0
  if grep -q "missing a model" "$A"; then
    echo "❌ 有配对缺模型,summary 是子集,不可用:"; grep "missing a model" "$A"; ok=0; fi
  tail -30 "$A"
  echo "== 逐病人"
  python -u scripts/subject_auc.py --runs "$OUT" --design results/matrix/design \
      --cohort "$COH" --replicates 1 --out "results/matrix/subject_auc/igfm_priv_$CELL" > "$B" 2>&1 || ok=0
  tail -20 "$B"
  # 打乱地板:PITFALLS 第 20 条。「26 个里有几个超过 0.55」的地板是 7 不是 0,
  # 而且地板是【每个 release 各算各的】,不能拿别的 arm 的地板来读。
  echo "== 打乱地板控制(PITFALLS 第 20 条)"
  python -u scripts/leak_locus.py --design "$DSN" --cohort "$COH" --runs "$OUT" \
      --out "results/matrix/leak_locus/igfm_priv_$CELL.json" > "$F" 2>&1 || ok=0
  tail -20 "$F"
  if [ "$ok" -eq 1 ]; then : > "$DONE"; echo "IGFM PRIV LOO $CELL 分析全部完成"
  else echo "❌ 分析失败(锁会在 trap 里归还,重排队会重试)"; rc=1; fi
elif [ "$n" -eq 27 ] && [ -f "$DONE" ]; then
  echo "27 个齐了,分析此前已完整跑过。"
elif [ "$n" -eq 27 ]; then
  echo "27 个齐了,但分析锁被别的作业占着。"
  # 持锁作业被硬杀时 trap 不执行,锁会变成死锁,而这条分支会让每次重排队都报绿。
  if [ ! -f "$ATK/summary.json" ]; then
    echo "❌ 而且 $ATK/summary.json 不存在 —— 可能已是死锁。确认无别的作业在跑后 rmdir $LOCK"; rc=1; fi
else
  echo "不足 27 个,不做分析 —— 少几个模型的攻击数字没有意义。重排队会接着跑。"; rc=1
fi
[ "$rc" -eq 0 ] || echo "== 本作业退出码 $rc,上面有失败项,别当成功读"
exit $rc
