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
# 2026-09-14 的扩充:七天的两个格子不在 25000 步训,而在【各自曲线选出的档位】训
# (d7_c1 = 12000,d7_c2 = 18000,后者 25000 步反而过训、Context-FID 从 0.2019 涨到
# 0.4205)。所以 steps 不再是常数,base 也不再是满预算那个 base,而是
# scripts/stage_base_from_curve.py 从同一档存档摆进来的那份。下面两处相应地改了:
#   * 逐键核对里的 steps 从写死的 25000 改成【从 PARAMS 里解析出来的那个】;
#   * 若 base 是摆进来的(meta 里有 budget_iters),额外核对它的档位 == steps。
# d1 的调用方一个字都不用改:不传 EXTRA 的 steps 就还是 25000,budget_iters 不存在
# 那一支就不走。
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
FREEZE=results/runs/.igfm_priv_campaign.params

BASEP=$(cat scripts/pbs/dev/igfm_priv.params | tr -d '\n' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')

# 战役级冻结,和 igfm_priv_d7_bases.pbs 用的是同一份。七天这批作业会在队列里排上
# 好几天,期间任何一次对共享参数文件的编辑都会让先跑的和后跑的 include 用不同参数
# 训,而【跨格子比较正是这个 arm 的全部意义】。下面那道 .params.json 指纹挡不住
# 这个:它只保证一棵树内部一致,四棵树各自漂各自的它一句话都不会说。
[ -f "$FREEZE" ] || { echo "❌ $FREEZE 不存在 —— d1 的战役应该已经冻结过它了"; exit 1; }
if [ "$(cat "$FREEZE")" != "$BASEP" ]; then
  echo "❌ 共享参数和本战役冻结的那份不一致,停。"
  echo "   冻住的: $(cat "$FREEZE")"; echo "   现在的: $BASEP"; exit 1
fi

PARAMS="${BASEP%\}}, $EXTRA}"
# steps / micro_batch 都从【解析后的 JSON】里取,不从 EXTRA 的文本里抠:JSON 里后
# 出现的键覆盖先出现的,所以 EXTRA 里的 steps 会盖掉 BASEP 的 25000,而文本匹配
# 看到的是两个。拼坏了在这里就崩,而不是四个分片各崩各的。
# 用 $(...) || 而不是 read <<EOF:后者在 python 失败时是【read 读到空、返回 1、
# set -e 把脚本掐掉】,下面那行自己写的错误提示永远不会被打印出来。
MB_STEPS=$(python - "$PARAMS" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
assert "micro_batch" in d, "params 缺 micro_batch"
assert int(d["steps"]) > 0, d["steps"]
print(int(d["micro_batch"]), int(d["steps"]))
PY
) || { echo "❌ params 拼坏了或缺键,上面是 python 的报错,停"; exit 1; }
read -r MB STEPS <<< "$MB_STEPS"
[ -n "${MB:-}" ] && [ -n "${STEPS:-}" ] || { echo "❌ 没能从 params 里解析出 micro_batch/steps"; exit 1; }
echo "[check] params 解析正确: micro_batch=$MB steps=$STEPS"

LOCK="$OUT/.analysis.lock"; DONE="$OUT/.analysis.done"; STAMP="$OUT/.params.json"
LOCK_HELD=0
mkdir -p logs "$OUT"

# base 是从 igfm_priv_<cell>/base 拷进来的实体(不是符号链接:共享目录的话,任何一次
# 往那边的重跑都会悄悄换掉 26 个配对的对照,而 subjects_sha1 检查照样通过)。
# 七天的两个格子则是 scripts/stage_base_from_curve.py 从曲线的同档里程碑摆进来的,
# 同样是实体拷贝,同样带一份 meta。
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

# ---- 守卫二:base 必须是本格子、带对模块、【同一个 micro_batch】、【同一个预算】的那个 base ----
[ -s "$OUT/base/samples.npy" ] || { echo "❌ $OUT/base/samples.npy 不在 —— 先把 base 摆进来(七天格子见 scripts/stage_base_from_curve.py)"; exit 1; }
WANT=$(grep -oE '"subjects_sha1": *"[0-9a-f]+"' "$OUT/base/meta.json" 2>/dev/null \
       | grep -oE '[0-9a-f]{16}' || true)
[ "$WANT" = "8cf77b7732b8fe79" ] || { echo "❌ base 不是本设计的 base(读到 ${WANT:-空}),停"; exit 1; }
grep -q "\"cohort\": \"$COH\"" "$OUT/base/meta.json" \
  || { echo "❌ base 不是本格子($COH)的,停"; exit 1; }
for kv in '"hidden": 144,' '"level_dims": 48,' '"order_dims": 48,' \
          "\"steps\": $STEPS," '"sampling_steps": 500,' \
          '"coord_mask": true,' '"p_block": 0.35,' '"p_level": 0.25,' \
          '"iso_weight": false,' "\"micro_batch\": $MB,"; do
  grep -q "$kv" "$OUT/base/meta.json" || { echo "❌ base 的 params 缺 [$kv],停"; exit 1; }
done
# 摆进来的 base 还要再验一道档位。params.steps 是 stage 脚本【改写】过的(它记的是
# 「这份权重等价于训 $STEPS 步的那次运行」),budget_iters 是同一件事的独立记录,
# 两者对不上就说明摆错了档,而上面那圈逐键核对会一声不吭地通过。
BI=$(grep -oE '"budget_iters": *[0-9]+' "$OUT/base/meta.json" | grep -oE '[0-9]+$' || true)
if [ -n "$BI" ]; then
  [ "$BI" = "$STEPS" ] || { echo "❌ base 是从曲线摆进来的,档位 $BI,而 include 要训 $STEPS 步 —— 这不是同预算的一对,停"; exit 1; }
  echo "base 是曲线里程碑 m$BI,与 include 同预算"
fi
echo "base 的 10 个关键参数全部对上(含 micro_batch=$MB, steps=$STEPS)"

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
  ) >> "logs/igfm_priv_loo_${CELL}_shard${s}.log" 2>&1 &
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

  # 成本核对。七天一个模型是 55.3 小时(c1 @12000)/ 80.5 小时(c2 @18000),而 g* 泳道
  # 的墙钟只有 24 小时,所以每个 include 都会续训 2-3 次 —— 而 run() 量的 fit_seconds
  # 只是【本次进程里那一段】。
  #
  # 这里【不能】再用「fit_seconds < 3600 就是续训」那个判据:那是给 base 写的,base 续
  # 训时走的是 start_it >= total_iters 的提前返回、真的只花几十秒;include 的最后一段
  # 是几小时到十几小时,判据一条都拦不住。两个 d7 base 的最后一段是 62 小时,它就是这样
  # 一声不响地穿过去、让整个战役预算错了 1.58 倍的(docs/PITFALLS.md §24)。
  #
  # 正确的字段现在由适配器自己给:fit_resumed / fit_seconds_total,loo/train.py 写进
  # meta。这里只做核对和汇总,不再猜。
  python - "$OUT" <<'PY' || ok=0
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
tot = 0.0; n = 0; resumed = 0; stale = []
for m in sorted(root.glob("*/meta.json")):
    d = json.loads(m.read_text())
    if d.get("role") == "base" or d.get("job") == "base":
        continue                       # base 是摆进来的,没有训练成本
    n += 1
    if d.get("fit_resumed"):
        resumed += 1
    t = d.get("fit_seconds_total")
    if t is None:
        # 这个模型是在适配器学会记累计墙钟之前训的,它的 fit_seconds 不可当成本读
        stale.append(m.parent.name)
        t = d.get("fit_seconds") or 0.0
    tot += float(t)
print(f"[cost] {n} 个 include,续训过的 {resumed} 个,"
      f"累计训练 {tot/3600:.1f} 卡时(均 {tot/3600/max(n,1):.1f} 小时/模型)")
if stale:
    print(f"[cost] ⚠️ {len(stale)} 个模型没有 fit_seconds_total,它们的 fit_seconds "
          f"只是最后一段,不能用来估预算:{stale[:5]}")
PY

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
