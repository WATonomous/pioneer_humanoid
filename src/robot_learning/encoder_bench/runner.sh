#!/usr/bin/env bash
# Job queue: runs the jobs in jobs.txt top to bottom, one at a time, each in a systemd scope capped at 56 GB
# (two jobs at once OOM-crashed the workstation on 2026-11-11). A job is "<tag> <command...>"; it is skipped once
# results/jobs/<tag>.done or .failed exists. jobs.txt is re-read after every job, so jobs can be appended at any time.
# When the list is exhausted the runner idles and checks again every minute.
cd "$(dirname "$0")"
export PY=.venv/bin/python B=ego+wrist_left
while true; do
  job=""
  while read -r tag cmd; do
    [[ -z "$tag" || "$tag" == \#* ]] && continue
    [[ -e results/jobs/$tag.done || -e results/jobs/$tag.failed ]] && continue
    job="$tag"; jobcmd="$cmd"; break
  done < jobs.txt
  if [[ -z "$job" ]]; then sleep 60; continue; fi
  echo "=== $job $(date +%T)"
  if systemd-run --user --scope -p MemoryMax=56G -p MemorySwapMax=2G --quiet bash -c "$jobcmd" > results/jobs/$job.log 2>&1; then
    touch results/jobs/$job.done; echo "    done $job $(date +%T)"
  else
    touch results/jobs/$job.failed; echo "    FAILED $job $(date +%T) (exit $?)"
  fi
  grep -hE "attn:|CLEAN|NOISY|LIGHT|COLOR|saved|Traceback|Killed" results/jobs/$job.log | cut -c1-200
done
