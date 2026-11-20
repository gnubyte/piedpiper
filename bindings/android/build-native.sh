#!/usr/bin/env bash
# Cross-compiles crates/jni (libpiedpiper_jni.so) for Android with the NDK's clang and copies
# the libraries into piedpiper/src/main/jniLibs/<abi>/, where the Gradle build picks them up.
#
#   ./build-native.sh                 # arm64-v8a, armeabi-v7a, x86_64
#   ./build-native.sh x86_64          # only the listed ABIs (arm64-v8a | armeabi-v7a | x86_64)
#
# Environment:
#   ANDROID_NDK_HOME / ANDROID_NDK_ROOT   NDK to use (default: newest under $ANDROID_HOME/ndk)
#   ANDROID_HOME / ANDROID_SDK_ROOT       SDK (default: local.properties sdk.dir, then the usual
#                                         per-OS install location)
#   ANDROID_API        minimum API level baked into the linker target (default 24 = minSdk)
#   CARGO_TARGET_DIR   cargo output directory (default: <repo>/target/android, kept apart from
#                      the desktop builds so concurrent cargo runs do not contend for a lock)
#   PIEDPIPER_STRIP    1 (default) strips the symbol table; 0 keeps it for native debugging
#
# Works with bash on Linux, macOS and Windows (Git Bash / MSYS2). Needs rustup.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
API="${ANDROID_API:-24}"
JNILIBS="$HERE/piedpiper/src/main/jniLibs"

case "$(uname -s)" in
    Linux*) HOST_TAG=linux-x86_64; EXE=""; WRAPPER="" ;;
    Darwin*) HOST_TAG=darwin-x86_64; EXE=""; WRAPPER="" ;;
    MINGW* | MSYS* | CYGWIN*) HOST_TAG=windows-x86_64; EXE=".exe"; WRAPPER=".cmd" ;;
    *) echo "build-native.sh: unsupported host $(uname -s)" >&2; exit 1 ;;
esac

# Paths handed to cargo/rustc must be native paths on Windows.
native_path() {
    if [ -n "$WRAPPER" ] && command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s\n' "$1"; fi
}

find_sdk() {
    local sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
    if [ -z "$sdk" ] && [ -f "$HERE/local.properties" ]; then
        # sdk.dir=C\:\\Users\\me\\AppData\\Local\\Android\\Sdk  ->  C:/Users/me/AppData/Local/Android/Sdk
        sdk="$(sed -n 's/^sdk\.dir=//p' "$HERE/local.properties" | sed -e 's/\\:/:/g' -e 's/\\\\/\//g' | tail -n 1)"
    fi
    if [ -z "$sdk" ]; then
        case "$HOST_TAG" in
            windows-*) sdk="${LOCALAPPDATA:-$HOME/AppData/Local}/Android/Sdk" ;;
            darwin-*) sdk="$HOME/Library/Android/sdk" ;;
            *) sdk="$HOME/Android/Sdk" ;;
        esac
    fi
    if [ -n "$WRAPPER" ] && command -v cygpath >/dev/null 2>&1; then sdk="$(cygpath -u "$sdk")"; fi
    printf '%s\n' "$sdk"
}

NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
if [ -z "$NDK" ]; then
    SDK="$(find_sdk)"
    if [ -d "$SDK/ndk" ]; then
        NDK="$(ls -1d "$SDK"/ndk/*/ 2>/dev/null | sort -V | tail -n 1)"
        NDK="${NDK%/}"
    fi
fi
if [ -n "$WRAPPER" ] && [ -n "$NDK" ] && command -v cygpath >/dev/null 2>&1; then NDK="$(cygpath -u "$NDK")"; fi
BIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin"
if [ -z "$NDK" ] || [ ! -d "$BIN" ]; then
    echo "build-native.sh: Android NDK not found (looked for '$BIN'); set ANDROID_NDK_HOME." >&2
    exit 1
fi

# ABI  rust-target  clang-target-prefix
ALL_ABIS="arm64-v8a armeabi-v7a x86_64"
abi_spec() {
    case "$1" in
        arm64-v8a) echo "aarch64-linux-android aarch64-linux-android" ;;
        armeabi-v7a) echo "armv7-linux-androideabi armv7a-linux-androideabi" ;;
        x86_64) echo "x86_64-linux-android x86_64-linux-android" ;;
        *) echo "build-native.sh: unknown ABI '$1' (expected one of: $ALL_ABIS)" >&2; return 1 ;;
    esac
}

ABIS=("$@")
if [ ${#ABIS[@]} -eq 0 ]; then
    read -r -a ABIS <<<"$ALL_ABIS"
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target/android}"
export CARGO_TARGET_DIR="$(native_path "$CARGO_TARGET_DIR")"
if [ "${PIEDPIPER_STRIP:-1}" = "1" ]; then
    export CARGO_PROFILE_RELEASE_STRIP=symbols
fi

AR="$(native_path "$BIN/llvm-ar$EXE")"
READELF="$BIN/llvm-readelf$EXE"
echo "NDK:        $NDK"
echo "API level:  $API"
echo "target dir: $CARGO_TARGET_DIR"

for abi in "${ABIS[@]}"; do
    spec="$(abi_spec "$abi")"
    target="${spec%% *}"
    clang_prefix="${spec##* }"
    linker="$BIN/${clang_prefix}${API}-clang${WRAPPER}"
    if [ ! -f "$linker" ]; then
        echo "build-native.sh: missing linker $linker (does this NDK support API $API?)" >&2
        exit 1
    fi
    if command -v rustup >/dev/null 2>&1 && ! rustup target list --installed | grep -qx "$target"; then
        echo "==> rustup target add $target"
        rustup target add "$target"
    fi
    upper="$(printf '%s' "$target" | tr 'a-z-' 'A-Z_')"
    lower="$(printf '%s' "$target" | tr '-' '_')"
    export "CARGO_TARGET_${upper}_LINKER=$(native_path "$linker")"
    export "CARGO_TARGET_${upper}_AR=$AR"
    # For crates that compile C code through the `cc` crate (none today, but cheap insurance).
    export "CC_${lower}=$(native_path "$linker")"
    export "AR_${lower}=$AR"

    echo "==> $abi ($target)"
    cargo build --release --manifest-path "$(native_path "$ROOT/crates/jni/Cargo.toml")" --target "$target"

    so="$CARGO_TARGET_DIR/$target/release/libpiedpiper_jni.so"
    if [ -n "$WRAPPER" ] && command -v cygpath >/dev/null 2>&1; then so="$(cygpath -u "$so")"; fi
    mkdir -p "$JNILIBS/$abi"
    cp "$so" "$JNILIBS/$abi/libpiedpiper_jni.so"

    # Android 15+ devices may use 16 KB pages: every LOAD segment must be 16 KB aligned.
    if [ -x "$READELF" ] || [ -f "$READELF" ]; then
        align="$("$READELF" -lW "$JNILIBS/$abi/libpiedpiper_jni.so" | awk '$1 == "LOAD" { print $NF }' | sort -u | tr '\n' ' ')"
        case "$abi" in
            arm64-v8a | x86_64)
                for a in $align; do
                    if [ $((a)) -lt 16384 ]; then
                        echo "warning: $abi LOAD segment alignment $a < 16 KB" >&2
                    fi
                done ;;
        esac
    fi
    echo "    -> $JNILIBS/$abi/libpiedpiper_jni.so ($(wc -c <"$JNILIBS/$abi/libpiedpiper_jni.so") bytes)"
done
