#!/bin/bash
set -e

echo "Waiting for frozen run (pid 3937085) to finish..."
tail --pid=3937085 -f /dev/null 2>/dev/null

echo "Frozen run done — killing script before Run 2 starts"
kill 3937083 2>/dev/null || true

export TBFM_DATA_DIR=/var/data/opto-coproc/
SESSIONS="MonkeyG_20150914_Session1_S1 MonkeyG_20150914_Session3_S1 MonkeyG_20150915_Session5_S1 MonkeyG_20150917_Session1_M1 MonkeyG_20150917_Session2_S1 MonkeyG_20150917_Session3_S1 MonkeyG_20150918_Session1_M1 MonkeyG_20150918_Session1_S1 MonkeyG_20150921_Session3_S1 MonkeyG_20150922_Session1_S1 MonkeyG_20150925_Session1_S1 MonkeyG_20150925_Session2_S1 MonkeyJ_20160426_Session1_S1 MonkeyJ_20160426_Session2_S1 MonkeyJ_20160429_Session1_S1 MonkeyJ_20160502_Session1_S1 MonkeyJ_20160627_Session1_S1 MonkeyJ_20160630_Session3_S1 MonkeyJ_20160702_Session2_S1 MonkeyJ_20160702_Session4_S1"

echo "=== Run 2: unfreeze bases only ==="
python tta_testing.py \
  --model-paths baseline:baseline \
  --support-sizes 1000 \
  --adapt-session $SESSIONS \
  --unfreeze-bases \
  --no-plot-display \
  --use-multi-gpu --gpu-ids 0 1 \
  --output-dir /home/danmuir/GitHub/py-tbfm/unfreeze_cmp/bases_20 \
  2>&1 | tee /home/danmuir/GitHub/py-tbfm/unfreeze_cmp/bases_20.log
