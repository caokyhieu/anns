#!/usr/bin/env bash
# Download, convert and verify the benchmark datasets against datasets.lock.json.
# Usage: scripts/prepare_data.sh [bigann|gist|all|verify]
# Files go to $ANNS_DATA (default <repo>/data). Existing verified files are kept.
# TLS verification stays on; pass a managed-network CA with BIGANN_CA_FILE / CURL_CA_BUNDLE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
: "${ANNS_DATA:=$ROOT/data}"
LOCK="$HERE/datasets.lock.json"
mkdir -p "$ANNS_DATA"

verify() {  # verify <dataset>
  python3 - "$LOCK" "$ANNS_DATA" "$1" <<'PY'
import hashlib, json, sys
from pathlib import Path
lock, root, name = json.load(open(sys.argv[1])), Path(sys.argv[2]), sys.argv[3]
spec, bad = lock[name], 0
for fname, want in spec["files"].items():
    path = root / spec["dir"] / fname
    if not path.is_file():
        print(f"  MISSING {path}"); bad += 1; continue
    if path.stat().st_size != want["bytes"]:
        print(f"  SIZE    {path}: {path.stat().st_size} != {want['bytes']}"); bad += 1; continue
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    ok = h.hexdigest() == want["sha256"]
    print(f"  {'ok     ' if ok else 'SHA256 '} {path}")
    bad += not ok
sys.exit(1 if bad else 0)
PY
}

bigann() {
  echo "== BIGANN 10M prefix -> $ANNS_DATA/bigann10m"
  if verify bigann10m; then return; fi
  python3 "$ROOT/native/scripts/prepare_bigann.py" --output "$ANNS_DATA/bigann10m" \
    ${BIGANN_CA_FILE:+--ca-file "$BIGANN_CA_FILE"}
  verify bigann10m
}

gist() {
  echo "== GIST1M -> $ANNS_DATA"
  if verify gist1m; then return; fi
  local h5="$ANNS_DATA/gist-960-euclidean.hdf5"
  if [ ! -f "$h5" ] || [ "$(stat -c %s "$h5")" != 3844648288 ]; then
    curl -fL --retry 3 -C - -o "$h5" "$(python3 -c "import json;print(json.load(open('$LOCK'))['gist1m']['url'])")"
  fi
  # convert_gist.py writes next to its input; h5py/numpy come from an ephemeral uv env.
  uv run --no-project --with h5py --with numpy python "$ROOT/lance_ivf_rq/scripts/convert_gist.py" "$h5"
  verify gist1m
}

case "${1:-all}" in
  bigann) bigann ;;
  gist) gist ;;
  all) bigann; gist ;;
  verify) verify bigann10m; verify gist1m ;;
  *) echo "usage: $0 [bigann|gist|all|verify]" >&2; exit 1 ;;
esac
