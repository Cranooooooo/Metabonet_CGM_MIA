# 加了隐私模块的 IG-FM 泄漏实验 —— 两个作业共用的正文。
#
# 调用方在 source 之前必须设好:SHARDS(本作业负责哪几个分片,空格分隔)
#
# ⚠️ 本文件是在 env.sh 之后被 source 的,而 env.sh 开头是 `set -euo pipefail`。
#    每一条会「正常地返回非零」的命令都必须显式兜住,否则整个作业在那一行悄悄退出。
#
# 分片总数固定 7:g3 拿 4 张卡跑 0-3,g2 拿 3 张卡跑 4-6。
# 27 个 job 切 7 份(strided slice,不重不漏,4/4/4/4/4/4/3),每份最多 4 个;
# 单模型实测 15715 秒训 + 830 秒采样 = 4.6 小时,4 个 = 18.4 小时,
# 24 小时墙钟留 5.6 小时余量(收尾分析约 0.5 小时)。
# 切 4 份的话每片 7 个 = 32 小时,超墙钟 —— 所以是 7 不是 4。

DSN=results/matrix/design/rep1
COH=data/cohort/matrix_d1_c1
OUT=results/runs/loo_igfm_priv_d1_c1
ATK=results/matrix/attack/igfm_priv_d1_c1
NS=7

# 和过了质量关的那个 base 逐字一致。iso_weight 保持 false:改动三已在 review 中撤出。
PARAMS='{"hidden":144,"level_dims":48,"order_dims":48,"steps":25000,"sampling_steps":500,"save_every":2000,"keep_checkpoints":2,"coord_mask":true,"p_block":0.35,"p_level":0.25,"iso_weight":false}'

LOCK="$OUT/.analysis.lock"
DONE="$OUT/.analysis.done"
STAMP="$OUT/.params.json"
LOCK_HELD=0

mkdir -p logs "$OUT"

# base 是从 igfm_priv_d1_c1/base 拷进来的(不是符号链接:共享一个目录的话,
# 任何一次往那边的重跑或 resample 都会把这 26 个配对的对照换掉,而
# subjects_sha1 检查照样通过 —— 那是看不见的失败)。来源仍记在 meta 的 workdir 里。
# 点号开头的 .analysis.lock / .params.json 不会被 */ 匹配到,不会污染计数。
count_done() {
  local n=0 d
  for d in "$OUT"/*/; do
    if [ -f "$d/samples.npy" ]; then n=$((n + 1)); fi
  done
  echo "$n"
}

# 唯一的 EXIT 处理。两件事:归还没用完的分析锁,报一句进度。
# 归还锁是必须的 —— 锁只在「干净地失败」时撤销的话,墙钟到点 / 节点撤离 /
# qdel 都会把它永久留在盘上,而之后每一次重排队都会看到 27 个模型齐、
# mkdir 失败、打印一句「已由另一个作业接手」、然后 exit 0。作业次次报绿,
# 攻击永远不跑,整批数据就这么埋了。
on_exit() {
  if [ "$LOCK_HELD" = "1" ] && [ ! -f "$DONE" ]; then
    rmdir "$LOCK" 2>/dev/null || true
    echo "== 分析未完成,已归还分析锁,重排队会重试"
  fi
  echo "== 收尾:$(count_done) / 27 个模型完成" || true
}
trap on_exit EXIT

# ---- 守卫一:参数漂移 ----------------------------------------------------
# 共用一个正文只保证「同一时刻两个作业一致」,不保证「今天训的 15 个和明天补训的
# 11 个一致」。run_loo 的跳过逻辑只比对 subjects_sha1,根本不看 params:改了
# PARAMS 再重排队,树里就会一半旧配置一半新配置,攻击照跑照出数,而
# 「有没有模块」这个唯一变量已经不唯一了。所以把参数拓在树上,对不上就停。
if [ -f "$STAMP" ]; then
  if [ "$(cat "$STAMP")" != "$PARAMS" ]; then
    echo "❌ PARAMS 和这棵树里已有模型用的不一致,停。"
    echo "   树里: $(cat "$STAMP")"
    echo "   现在: $PARAMS"
    echo "   换参数请换一个 --out 目录,不要在同一棵树里混两套配置。"
    exit 1
  fi
else
  printf '%s' "$PARAMS" > "$STAMP"
fi

# ---- 守卫二:base 必须是本设计、带对模块的那个 base ----------------------
# run_loo 自己也查 subjects_sha1,但要等它排到 base 那一格才查,而 base 在
# shard 0 里 —— 那时另外三张卡已经烧了几小时。这里先查,错了立刻停。
# 只查三个键不够:capacity 或步数不同的 base 能通过 sha1 检查,而那样
# include/base 的差别就混进了容量,不再只是隐私模块。逐个查。
WANT=$(grep -oE '"subjects_sha1": *"[0-9a-f]+"' "$OUT/base/meta.json" 2>/dev/null \
       | grep -oE '[0-9a-f]{16}' || true)
echo "base 的 subjects_sha1 = ${WANT:-读不到}"
[ "$WANT" = "8cf77b7732b8fe79" ] || { echo "❌ base 不是本设计的 base,停"; exit 1; }
for kv in '"hidden": 144' '"level_dims": 48' '"order_dims": 48' \
          '"steps": 25000' '"sampling_steps": 500' \
          '"coord_mask": true' '"p_block": 0.35' '"p_level": 0.25' \
          '"iso_weight": false'; do
  grep -q "$kv" "$OUT/base/meta.json" || { echo "❌ base 的 params 缺 [$kv],停"; exit 1; }
done
echo "base 的 9 个关键参数全部对上"

IFS=',' read -ra ALLOC <<< "${CUDA_VISIBLE_DEVICES:-0}"
NG=${#ALLOC[@]}
read -ra SH <<< "$SHARDS"
[ "$NG" -eq "${#SH[@]}" ] || { echo "❌ 拿到 $NG 张卡但要跑 ${#SH[@]} 个分片,停"; exit 1; }

# 注意:这一行其实是空转 —— env.sh 早就用 NCPUS 导出过 OMP/NUMBA/MKL_NUM_THREADS 了,
# 这里改 THREADS 不会重新导出它们,而且 Python 侧没有任何地方读 THREADS。
# 原版基线 igfm_loo.pbs 有一模一样的空转,所以两批模型的线程环境是同样的
# (各进程 OMP_NUM_THREADS=NCPUS,过订阅)。训练是 GPU-bound,实测无影响:
# 1 卡 16 CPU 的 base 训了 15715 秒,4 卡整机的分片是 15648-15715 秒。
# 明知是空转也保留,是为了不在两批之间引入任何一处不同。
export THREADS=$(( ${NCPUS:-16} / NG ))

echo "===== $(date '+%F %T')  IG-FM + 改动一 + 改动二 · 泄漏实验 ====="
echo "  分片 $SHARDS / $NS   卡 $NG 张   已完成 $(count_done) / 27"

pids=(); rc=0
for k in "${!SH[@]}"; do
  s="${SH[$k]}"
  ( export CUDA_VISIBLE_DEVICES="${ALLOC[$k]}"
    python -u scripts/run_loo.py --design "$DSN" --cohort "$COH" --out "$OUT" \
      --generator igfm --shard "$s" --n-shards "$NS" --seed 2026 \
      --params "$PARAMS"
  ) > "logs/igfm_priv_loo_shard${s}.log" 2>&1 &
  pids+=($!)
done
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then echo "分片 ${SH[$k]} OK"; else echo "分片 ${SH[$k]} 失败"; rc=1; fi
done

n=$(count_done)
echo "训练完成: $n / 27"

# 收尾靠「数够 27」而不是「我跑完了」:两个作业可能差几小时结束。
# mkdir 是原子的,同时数到 27 也只有一个能建成目录。
if [ "$n" -eq 27 ] && [ ! -f "$DONE" ] && mkdir "$LOCK" 2>/dev/null; then
  LOCK_HELD=1
  ok=1
  A=logs/igfm_priv_attack.log; B=logs/igfm_priv_subject_auc.log; F=logs/igfm_priv_floor.log

  echo "== 攻击(和原版 IG-FM 基线完全同一套冻结统计量,只有模型换了)"
  # 全量写进独立日志再 tail。run_attack 在某个配对缺模型时【不报错】:它打一句
  # 「N of 26 pairs are missing a model」到 stderr,跳过,然后照样写出 summary.json,
  # 里面的 AUC 是子集上算的。那句话在输出的最前面,tail -8 一定看不到。
  python -u scripts/run_attack.py --design "$DSN" --cohort "$COH" --runs "$OUT" \
      --out "$ATK" --generator igfm > "$A" 2>&1 || ok=0
  if grep -q "missing a model" "$A"; then
    echo "❌ 有配对缺模型,summary 是子集上算的,不可用:"; grep "missing a model" "$A"; ok=0
  fi
  tail -30 "$A"

  echo "== 逐病人"
  python -u scripts/subject_auc.py --runs "$OUT" --design results/matrix/design \
      --cohort "$COH" --replicates 1 \
      --out results/matrix/subject_auc/igfm_priv_d1_c1 > "$B" 2>&1 || ok=0
  tail -20 "$B"

  # 打乱地板。PITFALLS 第 20 条:把窗口内部打乱、销毁「哪条记录属于谁」,再跑
  # 完全相同的距离和逐病人 AUC。中位数会回到 0.503,但「26 个里有几个超过 0.55」
  # 这个计数的地板是 7,不是 0 —— 没有它,任何「计数从 N 降到 M」的句子都是错的,
  # 因为读者按 5% 假阳性去理解,实际是 27%。多跑一个变换,不额外训任何模型。
  echo "== 打乱地板控制(PITFALLS 第 20 条)"
  python -u scripts/leak_locus.py --design "$DSN" --cohort "$COH" --runs "$OUT" \
      --out results/matrix/leak_locus/igfm_priv_d1_c1 > "$F" 2>&1 || ok=0
  tail -20 "$F"

  if [ "$ok" -eq 1 ]; then
    : > "$DONE"
    echo "IGFM PRIV LOO 分析全部完成"
  else
    echo "❌ 分析这一步失败(锁会在 trap 里归还,重排队会重试)"
    rc=1
  fi
elif [ "$n" -eq 27 ] && [ -f "$DONE" ]; then
  echo "27 个齐了,分析此前已经完整跑过($DONE)。"
elif [ "$n" -eq 27 ]; then
  echo "27 个齐了,但分析锁被另一个作业占着。"
  # 这里不能无条件 exit 0。如果持锁的作业被硬杀(节点撤离、SIGKILL),
  # trap 不会执行,锁就成了死锁,而这条分支会让每次重排队都报绿。
  if [ ! -f "$ATK/summary.json" ]; then
    echo "❌ 而且 $ATK/summary.json 还不存在 —— 锁可能已经是死锁。"
    echo "   确认没有别的作业在跑之后,rmdir $LOCK 再排一次。"
    rc=1
  fi
else
  echo "不足 27 个,不做分析 —— 少几个模型的攻击数字没有意义。重排队会接着跑没做完的。"
  rc=1
fi

# rc=1 但分析跑成了的情况(某个分片在模型都写完之后才异常退出)也照实报,
# 不把它粉饰成成功。
[ "$rc" -eq 0 ] || echo "== 本作业退出码 $rc,上面有失败项,别当成功读"
exit $rc
