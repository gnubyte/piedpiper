#!/usr/bin/env bash
# Builds PiedPiper.app and packs it into dist/PiedPiper-<version>-macos.dmg, with an
# Applications shortcut to drag it onto.
#
#   bash apps/desktop/packaging/macos/build-dmg.sh [VERSION]
#
# Signing and notarization are optional. With MACOS_SIGN_IDENTITY set, the app and the DMG are
# signed with that Developer ID; with APPLE_ID, APPLE_TEAM_ID and APPLE_APP_PASSWORD (an
# app-specific password) also set, the DMG is notarized and stapled so Gatekeeper opens it
# without asking.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
app_dir="$(cd "$here/../.." && pwd)"
repo="$(cd "$app_dir/../.." && pwd)"

version="${1:-}"
if [ -z "$version" ]; then
  version="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$app_dir/Cargo.toml" | head -1)"
fi
version="${version#v}"
short="${version%%[-+]*}"

bash "$here/build-app.sh" "$version"

out_dir="${OUT_DIR:-$repo/dist}"
mkdir -p "$out_dir"
dmg="$out_dir/PiedPiper-$short-macos.dmg"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp -R "$app_dir/target/macos/PiedPiper.app" "$stage/"
ln -s /Applications "$stage/Applications"
rm -f "$dmg"
hdiutil create -volname "PiedPiper" -srcfolder "$stage" -fs HFS+ -format UDZO -ov "$dmg"

if [ -n "${MACOS_SIGN_IDENTITY:-}" ]; then
  codesign --force --timestamp --sign "$MACOS_SIGN_IDENTITY" "$dmg"
  if [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ] && [ -n "${APPLE_APP_PASSWORD:-}" ]; then
    xcrun notarytool submit "$dmg" --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" \
      --password "$APPLE_APP_PASSWORD" --wait
    xcrun stapler staple "$dmg"
  fi
fi
echo "$dmg"
