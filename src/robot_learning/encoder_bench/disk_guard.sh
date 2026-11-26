#!/bin/bash
# Pause both job queues if the root disk drops below 600 MB free (a full disk made every job fail on 2026-11-15).
cd "$(dirname "$0")"
while true; do
  free_mb=$(df -BM --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$free_mb" -lt 600 ] && [ ! -e results/PAUSE ]; then
    touch results/PAUSE; echo "$(date '+%F %T') disk guard: ${free_mb} MB free, queues paused" >> results/disk_cleanup.log
  fi
  sleep 60
done
