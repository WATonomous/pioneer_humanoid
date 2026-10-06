#!/usr/bin/env bash
# Fetch the Wato MJCF + meshes and the retargeted motion CSVs from the motion
# tracking repo (WATonomous/Humanoid_motion_tracking) into data/ (gitignored).
#
# The retargeted CSVs are joint angles for *that* repo's robot model (GMR's
# watonomous.xml, 28 joints), which differs from assets/whole_body_humanoid/
# in this repo, so the model and the motions are fetched together from one
# pinned commit.
#
#   ./scripts/fetch_assets.sh            # pinned commit
#   REV=<sha> ./scripts/fetch_assets.sh  # another commit
set -euo pipefail

REPO="https://github.com/WATonomous/Humanoid_motion_tracking"
REV="${REV:-4951224aa1766a302a7453196d52c2c41fbaebbd}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${HERE}/data"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$TMP" init -q
git -C "$TMP" remote add origin "$REPO"
git -C "$TMP" sparse-checkout set --no-cone \
    /GMR_Wato/assets/whole_body_humanoid/watonomous.xml \
    /GMR_Wato/assets/whole_body_humanoid/meshes/ \
    /GMR_Wato/retargeting_data/watonomous/wato_boxing.csv \
    /csv/
GIT_LFS_SKIP_SMUDGE=1 git -C "$TMP" fetch -q --depth 1 --filter=blob:none origin "$REV"
GIT_LFS_SKIP_SMUDGE=1 git -C "$TMP" checkout -q FETCH_HEAD

rm -rf "${DATA}/robot" "${DATA}/motions"
mkdir -p "${DATA}/robot" "${DATA}/motions"
cp "$TMP/GMR_Wato/assets/whole_body_humanoid/watonomous.xml" "${DATA}/robot/"
cp -r "$TMP/GMR_Wato/assets/whole_body_humanoid/meshes" "${DATA}/robot/"
cp "$TMP/GMR_Wato/retargeting_data/watonomous/wato_boxing.csv" "${DATA}/motions/boxing.csv"
cp "$TMP"/csv/*.csv "${DATA}/motions/"
echo "$REV" > "${DATA}/SOURCE_REV"

echo "Fetched ${REPO}@${REV:0:7} into ${DATA}:"
ls "${DATA}/motions"
