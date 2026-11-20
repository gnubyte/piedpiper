#!/usr/bin/env bash
# Installs the Mac app from a DMG and captures its windows for the README. Run on a GitHub
# macOS runner by .github/workflows/screenshots.yml; it copies the app into /Applications.
#
#   bash screenshots.sh PiedPiper-macOS.dmg OUTDIR
set -euo pipefail

dmg="$1"
out="$2"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
work="$(mktemp -d)"

mnt="$work/mnt"
mkdir -p "$mnt"
hdiutil attach "$dmg" -nobrowse -readonly -mountpoint "$mnt" >/dev/null
rm -rf /Applications/PiedPiper.app
cp -R "$mnt/PiedPiper.app" /Applications/
hdiutil detach "$mnt" -quiet
app=/Applications/PiedPiper.app
bin="$app/Contents/MacOS/PiedPiper"
# Real windows: no `ui = none` from an earlier test run.
rm -f "$HOME/Library/Application Support/PiedPiper/archiver.ini"

swiftc -O "$here/find_window.swift" -o "$work/find_window"

shoot() { # NAME SECONDS: wait for a PiedPiper window, let it settle, capture it without shadow
  local name="$1" settle="$2" id=""
  for _ in $(seq 1 60); do
    id="$("$work/find_window" PiedPiper 2>/dev/null || true)"
    [ -n "$id" ] && break
    sleep 0.5
  done
  if [ -z "$id" ]; then
    echo "no PiedPiper window for $name" >&2
    return 1
  fi
  sleep "$settle"
  screencapture -x -o -l "$id" "$out/$name.png"
  echo "captured $name (window $id)"
}

python3 - "$work/Meeting notes.txt" <<'PY'
import random, sys
random.seed(11)
words = ("the team agreed to ship the installer on friday after the review and to move the "
         "archive format notes into the design document so everyone can find them").split()
with open(sys.argv[1], "w") as f:
    f.write("\n".join(" ".join(random.choice(words) for _ in range(14)) for _ in range(9000)))
PY

# A compression in progress (stopped after the capture).
"$bin" compress "$work/Meeting notes.txt" &
pid=$!
shoot macos-progress 12 || true
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
sleep 1

# Opening the app: how to use it, and the compression level.
open -a "$app"
shoot macos-settings 3 || true
pkill -x PiedPiper || true

ls -la "$out"
