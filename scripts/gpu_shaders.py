#!/usr/bin/env python3
"""Compile PiedPiper's Vulkan kernels (crates/core/src/gpu/shaders/*.comp) to the SPIR-V in
crates/core/src/gpu/spv/. The SPIR-V is checked in, so building PiedPiper never needs a shader
compiler. Run this after changing a shader, then `cargo test -p piedpiper --features gpu`:
it checks that every float multiply, add and subtract carries NoContraction and that the
float-control execution modes are present, and (with a GPU) that the GPU model matches the CPU
bit for bit.

glslc comes with the Vulkan SDK or an Android NDK (shader-tools/<host>/glslc). Set GLSLC to its
path if it is not found automatically.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "crates" / "core" / "src" / "gpu" / "shaders"
OUT = ROOT / "crates" / "core" / "src" / "gpu" / "spv"


def find_glslc():
    exe = "glslc.exe" if os.name == "nt" else "glslc"
    if os.environ.get("GLSLC"):
        return os.environ["GLSLC"]
    on_path = shutil.which("glslc")
    if on_path:
        return on_path
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for sub in ("Bin", "bin"):
            p = Path(sdk) / sub / exe
            if p.exists():
                return str(p)
    bases = []
    for var in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT"):
        if os.environ.get(var):
            bases.append(Path(os.environ[var]))
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        if os.environ.get(var):
            bases.append(Path(os.environ[var]) / "ndk")
    if os.environ.get("LOCALAPPDATA"):
        bases.append(Path(os.environ["LOCALAPPDATA"]) / "Android" / "Sdk" / "ndk")
    home = Path.home()
    bases += [home / "Android" / "Sdk" / "ndk", home / "Library" / "Android" / "sdk" / "ndk"]
    for base in bases:
        roots = [base] + (sorted(base.iterdir(), reverse=True) if base.is_dir() else [])
        for r in roots:
            tools = r / "shader-tools"
            if tools.is_dir():
                for host in tools.iterdir():
                    p = host / exe
                    if p.exists():
                        return str(p)
    sys.exit("glslc not found: install the Vulkan SDK or an Android NDK, or set GLSLC")


def main():
    glslc = find_glslc()
    OUT.mkdir(exist_ok=True)
    wanted = set()
    for comp in sorted(SRC.glob("*.comp")):
        spv = OUT / (comp.stem + ".spv")
        wanted.add(spv.name)
        # -O0: the driver optimises anyway, and the SPIR-V stays a literal transcription of the
        # GLSL. No -g, so no file paths end up in the output.
        subprocess.run(
            [glslc, "--target-env=vulkan1.2", "-O0", "-I", str(SRC), "-o", str(spv), str(comp)],
            check=True,
        )
        print(f"{comp.name} -> {spv.relative_to(ROOT)}")
    for stale in OUT.glob("*.spv"):
        if stale.name not in wanted:
            stale.unlink()
            print(f"removed {stale.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
