#!/usr/bin/env bash
#
# Bootstrap the third-party sources needed to build race_auv_apriltag_cuda.
#
# The NVIDIA cuAprilTags static library lives in the isaac_ros_nitros
# repository as Git LFS objects. A plain clone would download every LFS
# blob in that repo (cuVSLAM, cuMotion, ... hundreds of MB). This script
# fetches the submodule, materializes only isaac_ros_nitros/lib/cuapriltags
# via a cone-mode sparse checkout, and pulls only the LFS objects in it.
#
# Usage:
#   scripts/setup_third_party.sh          # from a clean parent checkout
#   scripts/setup_third_party.sh --refresh  # re-run after a submodule move
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SM_REL="third_party/isaac_ros_nitros"
SM="${REPO_ROOT}/${SM_REL}"
CONE="isaac_ros_nitros/lib/cuapriltags"

command -v git-lfs >/dev/null 2>&1 || {
    echo "ERROR: git-lfs is required but not on PATH." >&2
    echo "       Install with 'sudo apt install git-lfs' or put the binary on PATH." >&2
    exit 1
}

# 1. Fetch the submodule. GIT_LFS_SKIP_SMUDGE keeps the checkout from
#    downloading the repository's unrelated LFS objects; the selected
#    cuapriltags blobs are pulled explicitly in step 3.
GIT_LFS_SKIP_SMUDGE=1 git -C "${REPO_ROOT}" submodule update --init "${SM_REL}"

# 2. Materialize only the cuapriltags subtree.
git -C "${SM}" sparse-checkout init --cone
git -C "${SM}" sparse-checkout set "${CONE}"
GIT_LFS_SKIP_SMUDGE=1 git -C "${SM}" checkout

# 3. Pull only the cuapriltags LFS objects.
git -C "${SM}" lfs pull --include="${CONE}/**"

LIB_DIR="${SM}/${CONE}/lib_aarch64_jetpack61/libcuapriltags.a"
if [ ! -s "${LIB_DIR}" ] || [ "$(stat -c %s "${LIB_DIR}")" -lt 10000 ]; then
    echo "ERROR: ${LIB_DIR} is missing or still an LFS pointer." >&2
    exit 1
fi

echo "cuapriltags ready: ${SM}/${CONE}"
