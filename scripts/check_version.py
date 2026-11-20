#!/usr/bin/env python3
"""Check that every package carries the same version, and that it matches a release tag.

    python scripts/check_version.py            # all package versions agree
    python scripts/check_version.py v0.1.2     # ... and equal the tag (CI does this for v* tags)

The versions live in five files: Cargo.toml ([workspace.package]: the CLI, the C library, the
Rust crates and the Python extension crate), apps/desktop/Cargo.toml, bindings/python/
pyproject.toml, bindings/js/package.json and bindings/android/app/build.gradle.kts
(versionName). Bump them together; also raise the Android versionCode.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def match(rel, pattern, flags=0):
    m = re.search(pattern, read(rel), flags)
    return m.group(1) if m else None


def versions():
    return {
        "Cargo.toml [workspace.package]": match(
            "Cargo.toml", r'^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"', re.M | re.S
        ),
        "apps/desktop/Cargo.toml": match("apps/desktop/Cargo.toml", r'^\[package\][^\[]*?^version\s*=\s*"([^"]+)"', re.M | re.S),
        "bindings/python/pyproject.toml": match(
            "bindings/python/pyproject.toml", r'^\[project\][^\[]*?^version\s*=\s*"([^"]+)"', re.M | re.S
        ),
        "bindings/js/package.json": json.loads(read("bindings/js/package.json")).get("version"),
        "bindings/android/app/build.gradle.kts": match(
            "bindings/android/app/build.gradle.kts", r'versionName\s*=\s*"([^"]+)"'
        ),
    }


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else None
    found = versions()
    width = max(map(len, found))
    for where, v in found.items():
        print(f"{where:<{width}}  {v}")
    problems = [f"{where}: no version found" for where, v in found.items() if not v]
    distinct = {v for v in found.values() if v}
    if len(distinct) > 1:
        problems.append("the packages disagree: " + ", ".join(sorted(distinct)))
    if tag is not None:
        want = tag[1:] if tag.startswith("v") else tag
        wrong = [where for where, v in found.items() if v and v != want]
        if wrong:
            problems.append(f"tag {tag} does not match: " + ", ".join(wrong))
    if problems:
        for p in problems:
            print(f"check_version: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"check_version: all packages are {distinct.pop()}" + (f", matching {tag}" if tag else ""))


if __name__ == "__main__":
    main()
