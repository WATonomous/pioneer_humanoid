#!/bin/bash
#SBATCH --job-name=peg-insert-stage
#SBATCH --partition=compute
#SBATCH --nodelist=trpro-slurm1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=tmpdisk:1024
#SBATCH --time=02:00:00
#SBATCH --output=/home/law_zou/humanoid-peg-insert/logs/stage-%j.out
set -euo pipefail
LOGIN=/home/law_zou/humanoid-peg-insert
PERSIST=/mnt/wato-drive/law_zou/humanoid-peg-insert
INPUT="$LOGIN/artifacts/input.tar.zst"
EXPECTED_SIZE=$1
EXPECTED_SHA=$2
printf 'Waiting for %s bytes of uploaded input\n' "$EXPECTED_SIZE"
for attempt in $(seq 1 600); do
  size=$(stat -c %s "$INPUT" 2>/dev/null || printf 0)
  if [ "$size" = "$EXPECTED_SIZE" ]; then
    printf '%s  %s\n' "$EXPECTED_SHA" "$INPUT" | sha256sum -c -
    if [ ! -e "$PERSIST" ] && [ -d /mnt/wato-drive/law_zou/peg_insert_20261013 ]; then
      mv /mnt/wato-drive/law_zou/peg_insert_20261013 "$PERSIST"
    fi
    mkdir -p "$PERSIST/artifacts" "$PERSIST/jobs" "$PERSIST/logs"
    cp "$INPUT" "$PERSIST/artifacts/input.tar.zst.part"
    printf '%s  %s\n' "$EXPECTED_SHA" "$PERSIST/artifacts/input.tar.zst.part" | sha256sum -c -
    mv "$PERSIST/artifacts/input.tar.zst.part" "$PERSIST/artifacts/input.tar.zst"
    cp "$LOGIN/jobs/"*.sbatch "$PERSIST/jobs/"
    # These are incomplete files from this run's canceled transfer; the verified input replaces them.
    rm -f "$PERSIST/input.tar.part"
    if [ -d "$PERSIST/datasets/peg_insert_raw" ]; then
      rm -r "$PERSIST/datasets/peg_insert_raw"
    fi
    rm "$INPUT"
    printf 'INPUT_STAGED_AND_VERIFIED\n'
    exit 0
  fi
  if [ $((attempt % 30)) = 0 ]; then printf 'Uploaded %s/%s bytes\n' "$size" "$EXPECTED_SIZE"; fi
  sleep 10
done
printf 'Upload did not complete within staging timeout\n' >&2
exit 1
