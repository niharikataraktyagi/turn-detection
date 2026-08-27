#!/usr/bin/env bash
# Downloads the AMI manual annotations (~10 MB). Word-level timings + speaker
# segments. This is all we need to build the TEXT dataset — no audio yet.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw
cd data/raw
URL="https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
if [ ! -f ami_public_manual_1.6.2.zip ]; then
  echo "downloading annotations..."
  curl -L -o ami_public_manual_1.6.2.zip "$URL"
fi
mkdir -p ami_annotations
unzip -o -q ami_public_manual_1.6.2.zip -d ami_annotations
echo "--- extracted ---"
du -sh ami_annotations
ls ami_annotations
echo
echo "words files:    $(ls ami_annotations/words 2>/dev/null | wc -l)"
echo "segments files: $(ls ami_annotations/segments 2>/dev/null | wc -l)"
