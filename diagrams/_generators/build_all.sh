#!/bin/bash
# Regenerate every .excalidraw file, then normalise text metrics and render previews.
set -e
cd "$(dirname "$0")"
for f in page*_*.py; do python3 "$f"; done
"$(dirname "$0")/harness/run.sh" ../page*/*.excalidraw
