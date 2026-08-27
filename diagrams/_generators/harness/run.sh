#!/bin/bash
# usage: ./run.sh <file.excalidraw>...
# One-time setup:
#   npm install
#   node_modules/.bin/esbuild entry.js  --bundle --format=iife --outfile=bundle.js  --define:process.env.NODE_ENV='"production"' --loader:.woff2=dataurl
#   node_modules/.bin/esbuild entry2.js --bundle --format=iife --outfile=bundle2.js --define:process.env.NODE_ENV='"production"' --loader:.woff2=dataurl
#   node genpage.js                       # writes page.html with the font faces
#   python3 -m http.server 8787 --bind 127.0.0.1 &
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
export HARNESS_URL="${HARNESS_URL:-http://127.0.0.1:8787}"
export PREVIEW_DIR="${PREVIEW_DIR:-$HERE/../../_preview}"
args=(); for a in "$@"; do args+=("$(readlink -f "$a")"); done
cd "$HERE"
node build.js "${args[@]}" 2>&1 | grep -vE "ERR_FILE|WorkerUrl|^    at |404 \(File not found\)|Failed to load resource"
