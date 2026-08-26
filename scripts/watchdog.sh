#!/bin/bash
# Cluster-side watchdog. Runs from the login-node crontab every 30 minutes.
#
#   crontab -e   ->   */30 * * * * <repo>/scripts/watchdog.sh
#
# WHY IT IS NOT A HUMAN AND NOT A CHAT SESSION. The campaign sat dead for 48 hours behind
# an `afterany` that released a failed prep, and the in-session watcher meant to catch it
# only fires when the chat REPL is idle, which it was not.
#
# WHAT WENT WRONG WITH THE FIRST VERSION, because the fixes below are shaped by it. On
# 08-22 it logged "relaunching the chain" three times. It relaunched nothing: the qsub it
# used to mint a dependency anchor carried no -P project, ASPIRE2A rejected it, stderr
# went to /dev/null, and `[ -n "$PREP" ] &&` swallowed the failure with no log line. It
# also wrote its one-shot marker BEFORE attempting, so three failed attempts left the
# watchdog permanently disarmed for a 108-model run. Every fix here is one of:
#   - never claim an action before it has succeeded
#   - never let a failed command look like a successful one
#   - never act on a reading you could not take
#
# It is not compute: qstat, ls and a couple of small reads, well under a second -- the
# same thing a person typing `qstat` does, so it does not breach
# NSCC_operation_guide_book.md section 1.
export PATH="/opt/pbs/bin:$PATH"
REPO=${REPO:-$HOME/workspace/Project_CGM/CGM-OutlierMIA-master}
export REPO                      # the status script must inspect the SAME tree
cd "$REPO" || exit 0
LOG=logs/watchdog.log
MARK=logs/.watchdog_relaunched
LOCK=logs/.watchdog.lock

# Two cron ticks can overlap if a qstat hangs, and the marker check-then-write is not
# atomic -- so both could pass it and both relaunch, giving two full 26-job chains on the
# same output directories. flock makes the whole run mutually exclusive; -n means the
# second one leaves rather than queues.
exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0

# a hung PBS server must not leave watchdogs piling up on the login node
OUT=$(timeout 120 bash scripts/matrix_status.sh 2>&1)
if [ $? -ne 0 ] && [ -z "$OUT" ]; then
  printf '%s  %-9s\n' "$(date '+%m-%d %H:%M')" "TIMEOUT" >> "$LOG"; exit 0
fi
V=$(printf '%s\n' "$OUT" | sed -n 's/^VERDICT: //p')
DONE=$(printf '%s\n' "$OUT" | sed -n 's|^  total \([0-9]*\)/.*|\1|p')
TOT=$(printf  '%s\n' "$OUT" | sed -n 's|^  total [0-9]*/\([0-9]*\).*|\1|p')
printf '%s  %-9s %s/%s\n' "$(date '+%m-%d %H:%M')" "${V%% *}" "${DONE:-?}" "${TOT:-?}" >> "$LOG"

case "$V" in
  UNKNOWN*)
    # The queue could not be read. That is not evidence of a stall, and acting on it is
    # how a watchdog relaunches on top of five healthy jobs.
    echo "  queue unreadable; no action" >> "$LOG"
    ;;
  STALLED*)
    if [ -f logs/.maintenance ]; then
      echo "  STALLED but logs/.maintenance is present: $(cat logs/.maintenance)" >> "$LOG"
      echo "  assuming deliberate; no relaunch" >> "$LOG"
      exit 0
    fi
    if [ -f "$MARK" ]; then
      echo "  already relaunched at $(cat "$MARK"); not doing it again" >> "$LOG"
      exit 0
    fi
    {
      echo "  STALLED -- full status:"
      printf '%s\n' "$OUT" | sed 's/^/    /'
      if [ ! -f results/matrix/design/rep1/design.json ]; then
        echo "  design MISSING -- prep never finished. NOT relaunching training;"
        echo "  the prep has to be fixed first. Last prep log lines:"
        tail -6 logs/B1c_rebuild.live.log 2>/dev/null | sed 's/^/    /'
        exit 0
      fi
      echo "  design exists, so this is a training stall; attempting relaunch"
      # launch_matrix.sh now takes no PREP: with nothing in flight there is nothing to
      # depend on, so round 1 just starts. That deletes the throwaway-anchor job whose
      # silent rejection is the reason no relaunch ever worked.
      if RES=$(bash scripts/pbs/launch_matrix.sh 2>&1); then
        # marker AFTER success, so a failed attempt does not burn the single allowance
        date '+%F %H:%M' > "$MARK"
        echo "  relaunch SUBMITTED:" ; printf '%s\n' "$RES" | tail -6 | sed 's/^/    /'
      else
        echo "  relaunch FAILED -- not marking, so the next tick may retry:"
        printf '%s\n' "$RES" | tail -8 | sed 's/^/    /'
      fi
    } >> "$LOG" 2>&1
    ;;
  COMPLETE*)
    grep -q "COMPLETE-noted" "$LOG" ||
      echo "  COMPLETE-noted: all ${TOT:-?} models done, analysis is due" >> "$LOG"
    ;;
esac
