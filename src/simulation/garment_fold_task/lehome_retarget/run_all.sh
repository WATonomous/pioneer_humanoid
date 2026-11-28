#!/bin/bash
# Retarget all 250 LeHome demos of one garment type (KIND=top_short|top_long|pant_short|pant_long) to the 8dof arm, one Isaac process per garment (25 demos each).
# Native Isaac Sim: ISAAC_PYTHON=<venv>/bin/python KIND=... ./run_all.sh
# Resumable: each garment's log_<a>_<b>.jsonl marks finished demos. Progress: out/$KIND/progress.txt
cd "$(dirname "$0")"
KIND=${KIND:-top_short}
OUT=${LEHOME_ROOT:-$HOME/Desktop/lehome_ref}/out/${OUTNAME:-$KIND}
mkdir -p $OUT
for gi in $(seq ${FIRST:-0} ${LAST:-9}); do
  a=$((gi * 25)); b=$((a + 25))
  if [ -n "$ISAAC_PYTHON" ]; then   # native Isaac Sim (pip) instead of the container
    LEHOME_ROOT=${LEHOME_ROOT:-$HOME/Desktop/lehome_ref} $ISAAC_PYTHON run_retarget.py --episodes $a:$b --kind $KIND --out $OUT --video 1 \
      --device cuda > $OUT/g$gi.log 2>&1
  else
    ./isaac_run.sh lehome_${KIND}_g$gi run_retarget.py --episodes $a:$b --kind $KIND --out /lehome/out/${OUTNAME:-$KIND} --video 1 --device cuda $EXTRA \
      > $OUT/g$gi.log 2>&1
  fi
  ok=$(cat $OUT/log_*.jsonl 2>/dev/null | grep -c '"success": true')
  n=$(cat $OUT/log_*.jsonl 2>/dev/null | grep -c '"ep"')
  echo "$(date +%H:%M) garment $gi done: $ok successful of $n tried" >> $OUT/progress.txt
done
echo "$(date +%H:%M) ALL DONE" >> $OUT/progress.txt
