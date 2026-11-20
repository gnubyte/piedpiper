#!/usr/bin/env python3
"""Write THIRD_PARTY_LICENSES.md: the license texts of every third-party crate compiled into a
PiedPiper binary (CLI, C library, Python wheel, WebAssembly, Android).

It lists the workspace's normal (non-build, non-dev) dependencies for all targets, skips
proc-macro crates (they run at compile time and are not linked) and PiedPiper's own crates, and
copies each crate's LICENSE / COPYING / NOTICE files from the local cargo registry. Crates that
ship no license file get the standard text of their declared license. Identical texts are
printed once and referenced afterwards. Run after `cargo fetch`:

    python scripts/gen_licenses.py

The desktop app (apps/desktop) is a workspace of its own, with platform GUI crates on top of the
core's dependencies, and ships its own copy:

    python scripts/gen_licenses.py --workspace apps/desktop --out apps/desktop/THIRD_PARTY_LICENSES.md
"""
import argparse
import glob
import hashlib
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = glob.glob(os.path.join(os.path.expanduser("~"), ".cargo", "registry", "src", "*"))
OWN = {"piedpiper", "piedpiper-cli", "piedpiper-capi", "piedpiper-jni", "piedpiper-wasm", "piedpiper-python",
       "piedpiper-media", "piedpiper-desktop", "cabac"}  # cabac = crates/cabac-shim, PiedPiper's own (MIT)
LICENSE_FILE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|UNLICENSE)", re.I)


def crates(workspace=ROOT):
    # --color never: CI sets CARGO_TERM_COLOR=always, which would wrap the " (*)" marker in escape codes.
    out = subprocess.run(["cargo", "tree", "--workspace", "-e", "normal", "--target", "all", "--prefix", "none",
                          "--color", "never", "--format", "{p}|{l}"],
                         cwd=workspace, capture_output=True, text=True, check=True).stdout
    seen = {}
    for line in out.splitlines():
        line = line.replace(" (*)", "").strip()
        if not line or "(proc-macro)" in line:
            continue
        pkg, lic = line.split("|", 1)
        parts = pkg.split()
        name, ver = parts[0], parts[1].lstrip("v")
        path = parts[2] if len(parts) > 2 else ""
        if name in OWN:
            continue
        seen[(name, ver)] = (lic.strip(), path)
    return sorted(seen.items())


def crate_dir(name, ver, path):
    if path.startswith("("):
        return path.strip("()")
    for r in REGISTRY:
        d = os.path.join(r, f"{name}-{ver}")
        if os.path.isdir(d):
            return d
    return None


def authors(d):
    try:
        toml = open(os.path.join(d, "Cargo.toml"), encoding="utf-8", errors="replace").read()
    except OSError:
        return ""
    m = re.search(r"^authors\s*=\s*\[(.*?)\]", toml, re.S | re.M)
    return ", ".join(re.findall(r'"([^"]+)"', m.group(1))) if m else ""


MIT = """Copyright (c) {who}

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def apache_text():
    return open(os.path.join(ROOT, "third_party", "lepton_jpeg", "LICENSE.txt"), encoding="utf-8").read()


def main():
    parser = argparse.ArgumentParser(description="Write the license texts of all third-party crates.")
    parser.add_argument("--workspace", default=ROOT, help="cargo workspace to list (default: the repository root)")
    parser.add_argument("--out", default=os.path.join(ROOT, "THIRD_PARTY_LICENSES.md"), help="file to write")
    opts = parser.parse_args()
    texts = {}  # sha -> (first crate, text)
    sections = []
    missing = []
    for (name, ver), (lic, path) in crates(opts.workspace):
        d = crate_dir(name, ver, path)
        files = []
        if d:
            for f in sorted(os.listdir(d)):
                full = os.path.join(d, f)
                if os.path.isfile(full) and LICENSE_FILE.match(f):
                    files.append((f, open(full, encoding="utf-8", errors="replace").read()))
        if not files:
            who = authors(d) if d else ""
            if "MIT" in lic:
                files.append(("MIT (standard text; the crate ships no license file)", MIT.format(who=who or f"the {name} authors")))
            elif "Apache-2.0" in lic:
                files.append(("Apache-2.0 (standard text; the crate ships no license file)", apache_text()))
            else:
                missing.append(f"{name} {ver} ({lic})")
        body = [f"## {name} {ver}", "", f"License: `{lic}`", ""]
        for fname, text in files:
            text = text.replace("\r\n", "\n").strip("\n")
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in texts:
                body.append(f"`{fname}`: identical to the text shown under {texts[h]}.")
                body.append("")
                continue
            texts[h] = f"{name} {ver}"
            body += [f"`{fname}`:", "", "```text", text, "```", ""]
        sections.append("\n".join(body))
    head = ["# Third-party licenses", "",
            "License texts of the third-party Rust crates compiled into PiedPiper's binaries, generated by",
            "`scripts/gen_licenses.py` from the workspace's dependency tree (see also",
            "[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Where a crate offers a choice of licenses,",
            "PiedPiper uses it under any one of them; all texts it ships are reproduced.", ""]
    if missing:
        head += ["Crates without a license text here: " + ", ".join(missing), ""]
    open(opts.out, "w", encoding="utf-8", newline="\n").write(
        "\n".join(head) + "\n" + "\n".join(sections))
    print(f"{len(sections)} crates, {len(texts)} distinct texts, missing: {missing}")


if __name__ == "__main__":
    main()
