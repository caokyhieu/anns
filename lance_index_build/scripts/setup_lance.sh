#!/usr/bin/env bash
# Create (or verify) worktrees/lance = upstream LANCE_BASE + patches/*.patch.
# The resulting git tree must equal LANCE_TREE, so x86 and ARM hosts build
# byte-identical sources. Usage: scripts/setup_lance.sh [--force]
#   LANCE_SRC=/path/to/existing/lance/clone  reuse a local clone as the fetch source.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/patches/SERIES.env"
WT="$HERE/worktrees/lance"
FORCE="${1:-}"

tree_of() { git -C "$WT" rev-parse 'HEAD^{tree}'; }

if [ -d "$WT/.git" ] || [ -f "$WT/.git" ]; then
  if [ -n "$(git -C "$WT" status --porcelain --untracked-files=no)" ]; then
    echo "worktrees/lance has uncommitted changes; commit or discard them first" >&2
    exit 1
  fi
  if [ "$(tree_of)" = "$LANCE_TREE" ]; then
    echo "worktrees/lance is at the expected tree $LANCE_TREE"
    exit 0
  fi
  if [ "$FORCE" != "--force" ]; then
    echo "worktrees/lance tree $(tree_of) != expected $LANCE_TREE." >&2
    echo "Rerun with --force to re-apply the patch series on a new branch." >&2
    exit 1
  fi
else
  mkdir -p "$HERE/worktrees"
  SRC="${LANCE_SRC:-$LANCE_URL}"
  echo "cloning $SRC (base $LANCE_BASE) into worktrees/lance"
  git init -q "$WT"
  git -C "$WT" remote add origin "$LANCE_URL"
  git -C "$WT" fetch --quiet --depth 1 "$SRC" "$LANCE_BASE"
fi

git -C "$WT" checkout -q -B "$LANCE_BRANCH" "$LANCE_BASE"
# Fixed identity/dates are irrelevant to the tree hash; only content is verified.
git -C "$WT" -c user.name=lance-index-build -c user.email=lance-index-build@localhost \
  am --quiet --keep-cr "$HERE"/patches/*.patch
if [ "$(tree_of)" != "$LANCE_TREE" ]; then
  echo "patched tree $(tree_of) != expected $LANCE_TREE" >&2
  exit 1
fi
echo "worktrees/lance ready on branch $LANCE_BRANCH (tree $LANCE_TREE)"
