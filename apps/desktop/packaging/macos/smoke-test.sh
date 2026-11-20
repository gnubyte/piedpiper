#!/usr/bin/env bash
# Checks a built DMG: mounts it, inspects the app, round-trips files with the bundled program,
# and (stage "services") installs the app and runs its Finder services the way the Finder does.
# Meant for CI; the services stage copies the app into /Applications.
#
#   bash smoke-test.sh app dist/PiedPiper-macOS.dmg
#   bash smoke-test.sh services dist/PiedPiper-macOS.dmg
set -euo pipefail

stage="$1"
dmg="$2"
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
mnt="$work/mnt"
mkdir -p "$mnt"
hdiutil attach "$dmg" -nobrowse -readonly -mountpoint "$mnt" >/dev/null
trap 'hdiutil detach "$mnt" -quiet || true' EXIT
app="$mnt/PiedPiper.app"
bin="$app/Contents/MacOS/PiedPiper"

wait_for() { # path, seconds
  for _ in $(seq 1 $(( $2 * 2 ))); do [ -e "$1" ] && return 0; sleep 0.5; done
  echo "timed out waiting for $1" >&2
  return 1
}

case "$stage" in
app)
  test -x "$bin"
  test -L "$mnt/Applications"
  plutil -lint "$app/Contents/Info.plist"
  codesign --verify --strict --verbose=2 "$app"
  archs="$(lipo -archs "$bin")"
  [[ "$archs" == *arm64* && "$archs" == *x86_64* ]] || { echo "not universal: $archs" >&2; exit 1; }
  for f in LICENSE.txt THIRD_PARTY_NOTICES.md THIRD_PARTY_LICENSES.md LICENSE-lepton_jpeg.txt NOTICE-lepton_jpeg.txt; do
    test -f "$app/Contents/Resources/Licenses/$f"
  done

  mkdir -p "$work/t/sub"
  for i in $(seq 1 200); do echo "line $i of a file to compress"; done > "$work/t/a.txt"
  echo inner > "$work/t/sub/b.txt"
  "$bin" --cli --level 1 compress "$work/t/a.txt" "$work/t/sub"
  test -f "$work/t/Archive.tar.pp"
  "$bin" --cli extract-to "$work/t/Archive.tar.pp"
  cmp "$work/t/a.txt" "$work/t/Archive/a.txt"
  cmp "$work/t/sub/b.txt" "$work/t/Archive/sub/b.txt"
  echo "App bundle OK; the bundled program round-trips files."
  ;;

services)
  rm -rf /Applications/PiedPiper.app
  cp -R "$app" /Applications/
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f /Applications/PiedPiper.app
  /System/Library/CoreServices/pbs -update || true
  settings="$HOME/Library/Application Support/PiedPiper"
  mkdir -p "$settings"
  printf 'level = 1\nlog = 1\nui = none\n' > "$settings/archiver.ini"
  show_log() { echo "--- archiver.log"; cat "$settings/archiver.log" 2>/dev/null || true; }
  trap 'show_log; rm -f "$settings/archiver.ini"; hdiutil detach "$mnt" -quiet || true' EXIT

  mkdir -p "$work/s/folder"
  for i in $(seq 1 100); do echo "service line $i"; done > "$work/s/one.txt"
  echo two > "$work/s/folder/two.txt"
  swift "$here/perform_service.swift" "Compress with PiedPiper" "$work/s/one.txt" "$work/s/folder"
  wait_for "$work/s/Archive.tar.pp" 120
  swift "$here/perform_service.swift" "Extract to Folder with PiedPiper" "$work/s/Archive.tar.pp"
  wait_for "$work/s/Archive/folder/two.txt" 120
  cmp "$work/s/one.txt" "$work/s/Archive/one.txt"
  echo "Finder services work end to end."
  ;;

*)
  echo "unknown stage $stage" >&2
  exit 2
  ;;
esac
