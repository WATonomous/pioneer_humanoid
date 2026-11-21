#!/usr/bin/env bash
# Wait until the shared workstation has room for a heavy job: >= $1 GB system RAM available and >= $2 GB GPU memory
# free. Starts on the first reading that has room (2026-11-13: the user asked not to wait more than needed). Other users share this machine (two big jobs at once crashed it on 2026-11-11).
#   ./gate.sh 40 8 && <job>
need_ram=$1; need_gpu=$2; ok=0; said=0
while [ $ok -lt 1 ]; do
  ram=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{printf "%d", $1/1024}')
  if [ "$ram" -ge "$need_ram" ] && [ "$gpu" -ge "$need_gpu" ]; then ok=$((ok+1)); else
    ok=0; [ $said -eq 0 ] && echo "$(date '+%T') gate: waiting for ${need_ram} GB RAM (have ${ram}) and ${need_gpu} GB GPU (have ${gpu})"; said=1; sleep 30; fi
done
echo "$(date '+%T') gate: room (RAM ${ram} GB, GPU ${gpu} GB free), starting"
