#!/bin/bash
# IG-FM + 2 模块,七天两个格子的 26 个 include —— 四条泳道、两个格子、一条链。
#
#   bash scripts/pbs/dev/launch_igfm_d7_loo.sh --dry-run     # 只打印 qsub,不提交
#   bash scripts/pbs/dev/launch_igfm_d7_loo.sh
#
# 泳道怎么来的。这台机器每个队列每人只能【同时跑一个作业】(max_run=
# [u:PBS_GENERIC=1]),每个节点四张卡,单节点作业最多四张,所以一个人能同时拿到的
# 上限就是 glong 4 + g3 4 + g2 3 + g1 1 = 12 张卡:
#
#   泳道 A  glong  4 卡 120 小时   分片 0 1 2 3
#   泳道 B  g3     4 卡  24 小时   分片 4 5 6 7
#   泳道 C  g2     3 卡  24 小时   分片 8 9 10
#   泳道 D  g1     1 卡  24 小时   分片 11
#
# 队列是按资源路由的,不是按 -q 选的:4 卡 + 120 小时 -> glong(它的墙钟下限是
# 24:00:01),4 卡 + 24 小时 -> g3(它要求恰好 4 卡),3 卡 -> g2,1 卡 -> g1。
#
# 为什么是「先把 d7_c1 十二张卡跑完,再跑 d7_c2」而不是一个格子一条泳道。
# 两个格子的总量是 1437 + 2093 = 3530 卡时。十二张卡都压在一个格子上,d7_c1 在
# 约 166 小时(最慢的分片 3 个模型)后整格可分析;按格子分泳道的话 d7_c2 只有
# glong 的四张卡,单是它就要 523 小时/卡。先窄后宽,总墙钟短一半,而且 d7_c1 的
# 攻击分析可以提前一周开始。
#
# 轮数怎么定的。单模型成本【从探针的 sec_per_iter 算,不从 meta 的 fit_seconds 算】:
# 14.9557 s/步(c1)和 15.0133 s/步(c2),加上实测 5.42 小时的采样。
#   c1 @12000 = 49.85 + 5.42 = 55.3 小时/模型
#   c2 @18000 = 75.07 + 5.43 = 80.5 小时/模型
# 两个 base 的 meta 里 fit_seconds 是 224491 秒,除以 25000 得 8.98 s/步 —— 那是错的,
# 它们都从 iter-10000 续过训,那 224491 秒只覆盖 15000 步。按它估会小 1.58 倍。
#   泳道 A(120 小时)  c1 最多 3x55.3 = 166 小时 -> 2 轮;c2 3x80.5 = 242 小时 -> 3 轮
#   泳道 BCD(24 小时) c1 2x55.3 = 111 小时 -> 5 轮,给 7;c2 2x80.5 = 161 小时 -> 7 轮,给 10
# BCD 按每轮有效 23 小时算(save_every 降到 500 之后,墙钟切掉一次最多丢 2.1 小时)。
# 多给的轮次不花钱:没活干的那一轮 run_loo 全部跳过,几秒钟就退出。少给才要命 ——
# 链断了得人工发现。
#
# 依赖一律 afterany,不是 afterok。一个分片失败不能把同泳道后面的都拖死:重排的
# 那一轮会跳过已完成的、接着训没完成的。PITFALLS 第 17 条讲的是反面 —— 用 afterok
# 挂在【前置准备】上,让失败的准备不要放行 26 个依赖;这里没有前置准备作业,base
# 是 igfm_d7_stage_base.pbs 提前摆好并在下面用文件存在性验过的。
set -euo pipefail
cd "$(dirname "$0")/../../.."
DRY=${1:-}
STAMP=$(date '+%m%d_%H%M')

# ---- 前提:两个 base 必须已经摆好 ----
# 训练作业里也验(而且验得更细:subjects_sha1、10 个参数键、budget_iters),但那时
# 已经排了几天队。这里只用 test,不碰 python(PITFALLS 第 7 条)。
for c in d7_c1 d7_c2; do
  for f in samples.npy meta.json; do
    [ -s "results/runs/loo_igfm_priv_$c/base/$f" ] || {
      echo "❌ results/runs/loo_igfm_priv_$c/base/$f 不在。" >&2
      echo "   先跑:qsub scripts/pbs/dev/igfm_d7_stage_base.pbs" >&2; exit 1; }
  done
done
echo "两个格子的配对 base 都在位"

# ---- 防重复:同名作业还在队列里就不要再发一条链 ----
QS=$(qstat -u "$(id -un)" 2>&1) || {
  echo "❌ qstat 读不到队列,无法确认有没有在跑的同名链 —— 拒绝盲发。" >&2
  echo "$QS" | tail -3 >&2; exit 1; }
if printf '%s\n' "$QS" | grep -qE " i7c[12]_"; then
  echo "❌ 队列里已经有 i7c*_ 作业,再发一条链会让两套作业往同一批目录里写。" >&2
  echo "   要重发先 qdel 干净:qselect -u \$(id -un) -N 'i7c*' | xargs -r qdel" >&2
  exit 1
fi

TALLY=$(mktemp); trap 'rm -f "$TALLY"' EXIT
sub(){ if [ "$DRY" = "--dry-run" ]; then { printf 'qsub'; printf ' %q' "$@"; printf '\n'; } >&2; echo "<job>"
       else qsub "$@"; fi; echo x >> "$TALLY"; }

# 泳道定义:名字 资源 墙钟 分片 c1轮数 c2轮数
LANES=(
  "A glong select=1:ncpus=64:ngpus=4:mem=440gb 120:00:00 0_1_2_3 2 3"
  "B g3    select=1:ncpus=64:ngpus=4:mem=440gb  24:00:00 4_5_6_7 7 10"
  "C g2    select=1:ncpus=48:ngpus=3:mem=330gb  24:00:00 8_9_10  7 10"
  "D g1    select=1:ncpus=16:ngpus=1:mem=110gb  24:00:00 11      7 10"
)
NS=12
EXPECT=0
for L in "${LANES[@]}"; do read -r _ _ _ _ _ r1 r2 <<< "$L"; EXPECT=$((EXPECT + r1 + r2)); done

for L in "${LANES[@]}"; do
  read -r lane queue sel wall shards r1 r2 <<< "$L"
  # shards 里已经是下划线形式,直接传给 -v(见泳道脚本里的还原)
  prev=""
  # 格子顺序就是链的顺序:d7_c1 的所有轮次跑完,才轮到 d7_c2。
  for spec in "d7_c1:$r1" "d7_c2:$r2"; do
    cell=${spec%%:*}; rounds=${spec##*:}
    for r in $(seq 1 "$rounds"); do
      short=${cell#d7_}                       # c1 / c2,作业名要短
      name="i7${short}_${lane}${r}"
      dep=""; [ -n "$prev" ] && dep="-W depend=afterany:$prev"
      id=$(sub -l "$sel" -l "walltime=$wall" \
           -v "CELL=$cell,SHARDS_U=$shards,NS=$NS" -N "$name" $dep \
           -o "logs/d7loo_${cell}_${lane}${r}_${STAMP}.log" \
           scripts/pbs/dev/igfm_priv_loo_d7_lane.pbs)
      echo "泳道 $lane($queue) $cell 第 $r 轮 -> $id"
      prev=${id%%.*}
    done
  done
done

# qsub 被拒时,set -e 会在 id=$(sub ...) 那一行就把脚本掐掉(命令替换子 shell 里
# 的 exit 状态会传出来),qsub 自己的 stderr 也没有被重定向,所以那是【响】的失败,
# 不是安静的 —— 下面这个计数拦不住它,也不是为它写的。它拦的是另一种:循环边界或
# 泳道表被改错,导致提交数和 EXPECT 对不上,而每一条 qsub 都成功了。
NSUB=$(wc -l < "$TALLY")
if [ "$NSUB" -ne "$EXPECT" ]; then
  echo "WARNING: 提交了 $NSUB 个作业,期望 $EXPECT —— 链是断的" >&2; exit 1
fi
echo "submitted $NSUB/$EXPECT"
[ "$DRY" = "--dry-run" ] || { echo; qstat -u "$(id -un)"; }
