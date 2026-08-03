#!/usr/bin/env bash
# Stop the attention monitor started by run_live.sh, and flush its mp4 cleanly.
#
# The bracketed pattern is deliberate: a plain `pkill -f attention.py` also matches
# the shell running the pkill, which kills your own session before it finishes.
set -u

PIDS=$(pgrep -f "[s]cripts/attention")
if [ -z "$PIDS" ]; then
  echo "not running"
  exit 0
fi

# SIGINT so the writer is released and the mp4 is playable
kill -INT $PIDS 2>/dev/null
for _ in $(seq 1 20); do
  sleep 0.5
  pgrep -f "[s]cripts/attention" > /dev/null || break
done
if pgrep -f "[s]cripts/attention" > /dev/null; then
  kill -TERM $(pgrep -f "[s]cripts/attention") 2>/dev/null
  sleep 1
fi
echo "stopped"
