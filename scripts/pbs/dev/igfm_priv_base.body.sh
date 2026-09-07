# IG-FM + 两个隐私模块,某一个格子的 base —— 三个格子共用的正文。
#
# 调用方必须设好(两个都是硬性的,漏了立刻停,不给默认值):
#   CELL   d1_c2 / d7_c1 / d7_c2
#   EXTRA  额外并进 params 的 JSON 片段(不含外层花括号)。必须包含 micro_batch:
#          七天窗口的 micro_batch 是探针量出来的,漏掉就会静默用适配器里写死的 32,
#          在 T=2016 上要么跑几小时后 OOM,要么训出一个和探针建议不一样的模型,
#          而唯一的痕迹只在 meta.json 里,没有任何东西会去比对它。
#
# base 先训、单独训,和 d1_c1 那次同样两个理由:
#   1) 它是质量关卡。两个模块都在往表示上加约束,而这三个格子的窗口长度和通道数
#      都和验证过的 d1_c1 不同 —— 在 T=288/C=1 上不赔质量,不等于在 T=2016 上不赔。
#   2) 它同时就是后面 27 个 MIA 模型里的【对照模型】,不是白训的一步。

# env.sh 开了 set -euo pipefail,但 conda activate 会 eval 一段生成的脚本并 source
# $ENV_PREFIX/etc/conda/activate.d/*.sh —— 里面任何一个做了 set +e,本文件唯一的
# 失败检测就没了:run_loo 可以死掉,后面两个打分步骤会写出空 JSON 并退 0,
# 作业最后照样打印 OK。所以这里既重新打开,也验一遍确实是开着的。
set -euo pipefail
case $- in *e*) ;; *) echo "❌ errexit 没打开,拒绝继续"; exit 1;; esac

: "${CELL:?CELL 必须设置}"
: "${EXTRA?EXTRA 必须设置(没有额外参数就设成空串)}"

DSN=results/matrix/design/rep1
COH="data/cohort/matrix_$CELL"
OUT="results/runs/igfm_priv_$CELL"
QOUT="results/igfm_priv_$CELL"
FREEZE=results/runs/.igfm_priv_campaign.params

# ${P%\}} 只在末尾正好是 } 时才截掉。文件尾多一个空格,截取就静默失效,
# 拼出来的是 `...false} , "micro_batch":8}` —— 目前 json 会报 Extra data 崩掉,
# 但那是靠 json 严格,不是靠这个脚本。先把首尾空白剥干净。
BASEP=$(cat scripts/pbs/dev/igfm_priv.params | tr -d '\n' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
P="${BASEP%\}}, $EXTRA}"
case "$P" in *'"micro_batch"'*) ;; *) echo "❌ params 里没有 micro_batch,停"; exit 1;; esac

mkdir -p logs "$QOUT"
LOG="logs/igfm_priv_base_$CELL.live.log"
trap 'tail -45 "'"$LOG"'" 2>/dev/null || true' EXIT

{
echo "===== $(date '+%F %T')  IG-FM + 改动一 + 改动二 · $CELL 的 base ====="
echo "参数: $P"

# ---- 守卫一:整个战役的共享参数,第一个作业跑起来的那一刻就冻住 --------------
# 共享一个参数文件只保证「同一时刻一致」。g1 每人只能跑一个作业、现在排着 125 个,
# 而 d7 两个格子还要等探针 —— 三个格子会隔着几小时到几天才轮到。这中间任何一次
# 对参数文件的编辑(比如为了绕开 d7 的采样 OOM 加个 sample_batch)都会静默地
# 改写还在排队的作业。不报错,唯一的痕迹在 meta.json,而没有东西去比对它。
# micro_batch 不在冻结范围内 —— 那是每个格子按窗口长度合法的差异。
if [ -f "$FREEZE" ]; then
  if [ "$(cat "$FREEZE")" != "$BASEP" ]; then
    echo "❌ 共享参数和本战役第一个作业冻住的那份不一致,停。"
    echo "   冻住的: $(cat "$FREEZE")"
    echo "   现在的: $BASEP"
    echo "   要换参数,请换一套输出目录另起一个战役。"
    exit 1
  fi
else
  printf '%s' "$BASEP" > "$FREEZE"
  echo "本战役的共享参数已冻结到 $FREEZE"
fi

# ---- 守卫二:本格子自己的参数指纹 ------------------------------------------
# run_loo 的跳过逻辑只比对 subjects_sha1,而它只来自 design —— 四个格子完全相同,
# 根本区分不了格子,更不看 params。适配器的续训也只认 workdir 里的 iter-*.pt,
# 不检查参数是否一致。所以「用 micro_batch=16 撞墙,改成 8 再排队」会从 18000 步
# 接着用新参数训完,不报错,meta.json 只记最后那个值。
STAMP="$OUT/.params.json"
mkdir -p "$OUT"
if [ -f "$STAMP" ]; then
  if [ "$(cat "$STAMP")" != "$P" ]; then
    echo "❌ 本格子已有模型用的参数和现在这套不一致,停。"
    echo "   已有: $(cat "$STAMP")"
    echo "   现在: $P"
    echo "   改了参数就换一个 --out 目录,不要在同一棵树里混两套配置"
    echo "   (要从头重训:先删掉 $OUT)。"
    exit 1
  fi
else
  printf '%s' "$P" > "$STAMP"
fi

python -u scripts/run_loo.py --design "$DSN" --cohort "$COH" --out "$OUT" \
    --generator igfm --job base --seed 2026 --params "$P"

# ---- 守卫三:确认真的训出东西了 --------------------------------------------
# eval_quality_tsgem 和 disc_stability 在找不到 samples.npy 时都是【打一行跳过、
# 写一个空 JSON、退 0】。那行提示在输出的最前面,而收尾只 tail 末尾几十行 ——
# 结果是空的质量文件 + 一句「OK」。d1_c1 那次有同样的洞,只是没踩上。
[ -s "$OUT/base/samples.npy" ] || { echo "❌ $OUT/base/samples.npy 不存在或为空,停"; exit 1; }
echo "训练产物: $(du -h "$OUT/base/samples.npy" | cut -f1)  $(ls "$OUT/base" | tr '\n' ' ')"

echo; echo "===== 质量打分:和 d1_c1 同一套口径 ====="
python -u scripts/eval_quality_tsgem.py --runs "$OUT/base" --cohort "$COH" \
    --design "$DSN" --out "$QOUT/tsgem.json" --label "igfm_priv_$CELL"
grep -q context_fid "$QOUT/tsgem.json" \
  || { echo "❌ $QOUT/tsgem.json 里没有 context_fid —— 打分实际上跳过了,停"; exit 1; }

echo; echo "===== 分真假:八次重启取最差,单次不算 ====="
# --job-file 必须给:不给的话脚本退回「拿整个队列当参照」,而我们用的是训练子集。
python -u scripts/disc_stability.py --runs "$OUT/base" --cohort "$COH" \
    --job-file "$DSN/jobs/base.json" --restarts 8 --out "$QOUT/disc_stability.json"

echo; echo "===== 本格子的 DiM-TS 对照 ====="
# 按【键】取,不是按位置取。原先 `grep context_fid | head -1` 拿的是文件里第一个,
# 那恰好是 base 只是因为写文件时 --runs 的顺序,换个顺序重跑就会把某个 include
# 模型的分数标成「DiM-TS 对照」,而且看不出来。
# 也【不要跨格子比绝对值】:每个格子的 DiM-TS 数不一样(d7_c2 的 base 是 0.290),
# 要比的是本格子里 IG-FM 对本格子的 DiM-TS。
for f in "results/matrix/quality_tsgem/${CELL}_allmodels.json" \
         "results/matrix/quality_tsgem/${CELL}.json"; do
  if [ -f "$f" ]; then
    v=$(grep -A20 "\"results/runs/matrix_$CELL/base\"" "$f" \
        | grep -oE '"context_fid": *[0-9.]+' | head -1 || true)
    echo "  DiM-TS($CELL) $v   <- 出自 $f"
    break
  fi
done
echo "  IG-FM+模块($CELL) $(grep -oE '"context_fid": *[0-9.]+' "$QOUT/tsgem.json" | head -1)"

echo; echo "===== IGFM PRIV BASE $CELL OK $(date '+%F %T') ====="
} > "$LOG" 2>&1
tail -45 "$LOG"
