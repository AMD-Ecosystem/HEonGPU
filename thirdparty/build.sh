#!/usr/bin/env bash
set -e

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

git submodule update --init --recursive

# GPU-NTT, GPU-FFT and RNGonGPU are separate upstream repositories pinned here as
# submodules, so their AMD GPU support cannot live in their checked-out tree. It
# is carried as patches applied on top of the pinned commit. Each patch is a
# no-op on NVIDIA builds; applying is skipped when the patch is already in place
# so repeated configures stay idempotent.
if [ "${1:-OFF}" = "ON" ]; then
    for name in GPU-NTT GPU-FFT RNGonGPU; do
        patch="$here/patches/$name.patch"
        [ -f "$patch" ] || continue
        if git -C "$here/$name" apply --reverse --check "$patch" 2>/dev/null; then
            continue
        fi
        git -C "$here/$name" apply "$patch"
    done
fi
