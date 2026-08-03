#!/usr/bin/env bash
# Launch the attention monitor on the board's physical display, for filming.
#
#   ./scripts/run_live.sh [output.mp4]
#
# Opens on a live preview. Look at your target and press c to calibrate, r to
# recalibrate at any time, q or Esc to quit.
#
# Runs detached, so closing the SSH session does not kill it. Stop it with
# scripts/stop_live.sh, or press q / Esc in the window.
set -u
cd "$(dirname "$0")/.."

OUT="${1:-results/live.mp4}"
LOG=/tmp/demo.log

XAUTH=$(ls /run/user/1000/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
if [ -z "$XAUTH" ]; then
  echo "no Xwayland auth file; is the desktop session up?" >&2
  exit 1
fi
if [ ! -e /dev/video0 ]; then
  echo "no /dev/video0; is the webcam plugged in?" >&2
  exit 1
fi

rm -f "$LOG"
DISPLAY=:0 XAUTHORITY="$XAUTH" setsid \
  ./venv/bin/python -u scripts/attention.py \
    --source 0 --display --calibrate \
    --calib-seconds 4 \
    --alert-after 2.0 \
    -o "$OUT" \
  > "$LOG" 2>&1 < /dev/null &

sleep 3
if pgrep -f "[s]cripts/attention" > /dev/null; then
  echo "running (pid $(pgrep -f '[s]cripts/attention' | head -1))"
  echo "  press c in the window to calibrate (4 s), r to redo, q to quit"
  echo "  recording : $OUT"
  echo "  log       : $LOG"
else
  echo "failed to start; last lines of $LOG:" >&2
  tail -5 "$LOG" >&2
  exit 1
fi
