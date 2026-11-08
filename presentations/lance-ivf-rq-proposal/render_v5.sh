#!/usr/bin/env bash
# Render the v5 deck to PDF + per-slide PNG + contact sheet.
# Uses system libreoffice/pdftoppm if present, else the locally extracted copy in /tmp/ivfrq-render.
set -euo pipefail
cd "$(dirname "$0")"
DECK=${1:-ads-lance-ivf-rq-proposal-v5.pptx}
OUT=${2:-preview-v5}
mkdir -p "$OUT"
if command -v soffice >/dev/null; then SOFFICE=soffice; PDFTOPPM=pdftoppm; else
  R=/tmp/ivfrq-render/root
  export LD_LIBRARY_PATH=$R/usr/lib/x86_64-linux-gnu:$R/usr/lib/libreoffice/program:$R/lib/x86_64-linux-gnu
  SOFFICE=$R/usr/lib/libreoffice/program/soffice; PDFTOPPM=$R/usr/bin/pdftoppm
  PROFILE=(-env:UserInstallation=file:///tmp/ivfrq-render/profile)
fi
"$SOFFICE" ${PROFILE[@]+"${PROFILE[@]}"} --headless --convert-to pdf --outdir "$OUT" "$DECK" >/dev/null
rm -f "$OUT"/slide-*.png
"$PDFTOPPM" -r 110 -png "$OUT/${DECK%.pptx}.pdf" "$OUT/slide"
for f in "$OUT"/slide-*.png; do n=${f##*-}; n=${n%.png}; t="$OUT/slide-$(printf %02d $((10#$n))).png"; [ "$f" = "$t" ] || mv "$f" "$t"; done
python - "$OUT" <<'EOF'
import sys, glob
from PIL import Image
files = sorted(glob.glob(sys.argv[1] + '/slide-*.png'))
ims = [Image.open(f) for f in files]
w, h = ims[0].size; tw, th = w // 2, h // 2; cols = 3; rows = (len(ims) + cols - 1) // cols
sheet = Image.new('RGB', (cols * tw, rows * th), 'white')
for i, im in enumerate(ims):
    sheet.paste(im.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
sheet.save(sys.argv[1] + '/contact-sheet.png')
EOF
echo "rendered $(ls "$OUT"/slide-*.png | wc -l) slides to $OUT"
