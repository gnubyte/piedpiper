#!/usr/bin/env bash
# Builds apps/desktop/target/macos/PiedPiper.app, a universal (Apple Silicon + Intel) app.
#
#   bash apps/desktop/packaging/macos/build-app.sh [VERSION]
#
# Signs with Developer ID when MACOS_SIGN_IDENTITY names a certificate in the keychain
# (hardened runtime, as notarization requires); otherwise signs ad hoc, which runs locally
# but makes Gatekeeper ask first on other Macs.
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

export MACOSX_DEPLOYMENT_TARGET=11.0
for target in aarch64-apple-darwin x86_64-apple-darwin; do
  cargo build --release --locked --manifest-path "$app_dir/Cargo.toml" --target "$target"
done

bundle="$app_dir/target/macos/PiedPiper.app"
rm -rf "$bundle"
mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources/Licenses"
lipo -create -output "$bundle/Contents/MacOS/PiedPiper" \
  "$app_dir/target/aarch64-apple-darwin/release/piedpiper-archiver" \
  "$app_dir/target/x86_64-apple-darwin/release/piedpiper-archiver"
sed "s/@VERSION@/$short/g" "$here/Info.plist" > "$bundle/Contents/Info.plist"
plutil -lint "$bundle/Contents/Info.plist"
printf 'APPL????' > "$bundle/Contents/PkgInfo"
cp "$app_dir/resources/macos/AppIcon.icns" "$app_dir/resources/macos/ArchiveIcon.icns" "$bundle/Contents/Resources/"

licenses="$bundle/Contents/Resources/Licenses"
cp "$repo/LICENSE" "$licenses/LICENSE.txt"
cp "$repo/THIRD_PARTY_NOTICES.md" "$app_dir/THIRD_PARTY_LICENSES.md" "$licenses/"
cp "$repo/third_party/lepton_jpeg/LICENSE.txt" "$licenses/LICENSE-lepton_jpeg.txt"
cp "$repo/third_party/lepton_jpeg/NOTICE.txt" "$licenses/NOTICE-lepton_jpeg.txt"

if [ -n "${MACOS_SIGN_IDENTITY:-}" ]; then
  codesign --force --options runtime --timestamp --sign "$MACOS_SIGN_IDENTITY" "$bundle"
else
  codesign --force --sign - "$bundle"
fi
codesign --verify --strict --verbose=2 "$bundle"
echo "$bundle"
