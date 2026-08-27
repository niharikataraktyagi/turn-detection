#!/usr/bin/env bash
# AMI headset (IHM) audio for a chosen set of meetings.
# ~230 MB per meeting (4 headsets). Run in YOUR terminal — long-running.
#   nohup bash scripts/02_download_audio.sh > data/raw/audio_dl.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
OUT=data/raw/ami_audio
mkdir -p "$OUT"
BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

# 10 meetings spanning 6 different speaker groups (ES/IS/TS sites) so that we
# can hold out WHOLE GROUPS at split time and never leak a speaker across it.
MEETINGS="ES2002a ES2002b ES2003a ES2003b IS1000a IS1001a IS1003b TS3003a TS3004a TS3005a"

for m in $MEETINGS; do
  for h in 0 1 2 3; do
    f="$OUT/${m}.Headset-${h}.wav"
    [ -s "$f" ] && { echo "skip $(basename $f)"; continue; }
    url="$BASE/$m/audio/${m}.Headset-${h}.wav"
    echo "GET $url"
    curl -fsSL --retry 3 -o "$f" "$url" || { echo "  MISSING (ok if <4 speakers)"; rm -f "$f"; }
  done
done
echo "=== done ==="
du -sh "$OUT"; ls -la "$OUT" | head -50
