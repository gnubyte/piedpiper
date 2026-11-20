#!/usr/bin/env python3
"""Copy PiedPiper's license and third-party notices into the npm and Python packages.

npm and maturin only pack files inside their own project directories, so bindings/js and
bindings/python each keep a copy of the notice files listed below. The Android library copies
them at build time instead (bindings/android/piedpiper/build.gradle.kts).

bindings/js/build.py refreshes the npm copies on every build. Run this yourself after editing
LICENSE or THIRD_PARTY_NOTICES.md, or after regenerating THIRD_PARTY_LICENSES.md
(scripts/gen_licenses.py):

    python bindings/sync_notices.py           # refresh the copies in bindings/js and bindings/python
    python bindings/sync_notices.py --check   # only report missing or stale copies (exit status 1)
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Repository-root file -> its copy, per package directory under bindings/. Keep in step with
# "files" in bindings/js/package.json, license-files in bindings/python/pyproject.toml and
# noticePaths in bindings/android/piedpiper/build.gradle.kts.
# The npm copies keep the repository layout, so the relative links in THIRD_PARTY_NOTICES.md work
# inside the package. The Python copies sit side by side: maturin moves license files to the root
# of the source distribution, where the vendored third_party/lepton_jpeg crate already has its own
# LICENSE.txt and NOTICE.txt, and identical paths would make `maturin sdist` fail.
NOTICES = {
    "js": {
        "LICENSE": "LICENSE",
        "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
        "THIRD_PARTY_LICENSES.md": "THIRD_PARTY_LICENSES.md",
        "third_party/lepton_jpeg/LICENSE.txt": "third_party/lepton_jpeg/LICENSE.txt",
        "third_party/lepton_jpeg/NOTICE.txt": "third_party/lepton_jpeg/NOTICE.txt",
    },
    "python": {
        "LICENSE": "LICENSE",
        "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
        "THIRD_PARTY_LICENSES.md": "THIRD_PARTY_LICENSES.md",
        "third_party/lepton_jpeg/LICENSE.txt": "LICENSE-lepton_jpeg.txt",
        "third_party/lepton_jpeg/NOTICE.txt": "NOTICE-lepton_jpeg.txt",
    },
}


def read(path):
    with open(path, "rb") as f:
        return f.read()


def shown(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def require_sources():
    for src in NOTICES["js"]:
        if not os.path.isfile(os.path.join(ROOT, src)):
            hint = " (generate it with: python scripts/gen_licenses.py)" if src == "THIRD_PARTY_LICENSES.md" else ""
            sys.exit(f"sync_notices: missing {os.path.join(ROOT, src)}{hint}")


def stale(package):
    """(original, copy) path pairs of a package whose copy is missing or out of date."""
    dest = os.path.join(ROOT, "bindings", package)
    pairs = [(os.path.join(ROOT, src), os.path.join(dest, dst)) for src, dst in NOTICES[package].items()]
    return [(src, dst) for src, dst in pairs if not os.path.isfile(dst) or read(dst) != read(src)]


def copy_notices(package):
    """Refresh the copies of one package ("js" or "python"), leaving up-to-date ones untouched."""
    require_sources()
    for src, dst in stale(package):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"sync_notices: wrote {shown(dst)}")


def main():
    if "--check" in sys.argv[1:]:
        require_sources()
        bad = [shown(dst) for package in NOTICES for _, dst in stale(package)]
        for path in bad:
            print(f"sync_notices: missing or stale: {path}")
        sys.exit(1 if bad else 0)
    for package in NOTICES:
        copy_notices(package)


if __name__ == "__main__":
    main()
