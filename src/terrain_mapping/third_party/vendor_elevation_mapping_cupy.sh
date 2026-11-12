#!/usr/bin/env bash
# Re-vendor upstream elevation_mapping_cupy (ros2 branch) at UPSTREAM_COMMIT, apply patches/*.patch,
# and drop upstream's tests.
# Upstream targets Jazzy; the patches are what it takes to run on Humble.
set -euo pipefail

UPSTREAM_URL=https://github.com/leggedrobotics/elevation_mapping_cupy.git
UPSTREAM_COMMIT=d9e1c9fe42ea3d463519333a443da99f680adde8

here="$(cd "$(dirname "$0")" && pwd)"
dest="$here/elevation_mapping_cupy"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

git clone -q --filter=blob:none --no-checkout "$UPSTREAM_URL" "$tmp/upstream"
git -C "$tmp/upstream" checkout -q "$UPSTREAM_COMMIT"

rm -rf "$dest"
mkdir -p "$dest"
cp -r "$tmp/upstream/LICENSE" "$tmp/upstream/elevation_mapping_cupy" "$tmp/upstream/elevation_map_msgs" "$dest/"

shopt -s nullglob
for p in "$here"/patches/*.patch; do
  patch -p1 -d "$dest" --forward --quiet < "$p"
  echo "applied $(basename "$p")"
done
rm -rf "$dest/elevation_mapping_cupy/test" "$dest/elevation_mapping_cupy/elevation_mapping_cupy/tests" \
  "$dest/elevation_mapping_cupy/elevation_mapping_cupy/listener_test.py"
echo "vendored $UPSTREAM_COMMIT into $dest"
