#!/usr/bin/env bash
# GitHub release steps for the workflows (needs the gh CLI and GH_TOKEN).
#
#   release.sh ensure  TAG          create the release as a pre-release, with the download notes
#                                   from .github/release-notes.md, unless it already exists
#   release.sh upload  TAG FILE...  attach files (same-named ones are replaced)
#   release.sh publish TAG          make it the latest full release
#
# A release stays a pre-release until the desktop workflow has attached the tested Windows MSI
# and macOS DMG and calls `publish`, so "Latest" on GitHub, and the
# releases/latest/download/... links in the README, always have the installers.
set -euo pipefail

cmd="$1"
tag="$2"
shift 2
version="${tag#v}"
root="$(cd "$(dirname "$0")/.." && pwd)"

ensure() {
  if gh release view "$tag" >/dev/null 2>&1; then
    return
  fi
  local notes
  notes="$(mktemp)"
  sed "s/{{VERSION}}/$version/g" "$root/.github/release-notes.md" > "$notes"
  # Another job may create it at the same moment; that is fine as long as it exists afterwards.
  gh release create "$tag" --verify-tag --prerelease --title "PiedPiper $version" \
    --notes-file "$notes" --generate-notes || gh release view "$tag" >/dev/null
  rm -f "$notes"
}

case "$cmd" in
  ensure)
    ensure
    ;;
  upload)
    ensure
    gh release upload "$tag" "$@" --clobber
    ;;
  publish)
    gh release edit "$tag" --prerelease=false --latest
    ;;
  *)
    echo "usage: release.sh ensure|upload|publish TAG [FILE...]" >&2
    exit 2
    ;;
esac
