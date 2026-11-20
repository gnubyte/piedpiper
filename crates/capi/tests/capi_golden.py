#!/usr/bin/env python3
"""Exercises the C ABI through ctypes (no C compiler needed): golden vectors, progress,
cancellation, inspect and priors.

usage: python crates/capi/tests/capi_golden.py [path/to/libpiedpiper.{so,dylib,dll}]
"""
import ctypes as C
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GOLDEN = os.path.join(ROOT, "testdata", "golden")


def default_lib():
    name = {"win32": "piedpiper.dll", "darwin": "libpiedpiper.dylib"}.get(sys.platform, "libpiedpiper.so")
    return os.path.join(ROOT, "target", "release", name)


class Buffer(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_uint8)), ("len", C.c_size_t)]


class Model(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("d_model", "n_layer", "n_head", "d_ff", "window", "seg_len", "n_streams")]


class Train(C.Structure):
    _fields_ = [("lr", C.c_float), ("lr_min", C.c_float), ("warmup", C.c_uint32), ("decay", C.c_uint32),
                ("beta1", C.c_float), ("beta2", C.c_float), ("eps", C.c_float), ("clip", C.c_float),
                ("init_scale", C.c_float), ("seed", C.c_uint64)]


class Options(C.Structure):
    _fields_ = [("level", C.c_uint32), ("tokenizer", C.c_uint32), ("bpe_vocab", C.c_uint32),
                ("threads", C.c_uint32), ("allow_stored", C.c_uint32), ("has_model", C.c_uint32),
                ("model", Model), ("has_train", C.c_uint32), ("train", Train), ("builtin_prior", C.c_uint32)]


PROGRESS = C.CFUNCTYPE(C.c_int, C.c_uint64, C.c_uint64, C.c_void_p)


def main():
    lib = C.CDLL(sys.argv[1] if len(sys.argv) > 1 else default_lib())
    lib.pp_version.restype = C.c_char_p
    lib.pp_last_error.restype = C.c_char_p
    lib.pp_prior_load.restype = C.c_void_p
    lib.pp_prior_id.restype = C.c_uint64
    lib.pp_compress.argtypes = [C.c_char_p, C.c_size_t, C.POINTER(Options), C.c_void_p, PROGRESS, C.c_void_p, C.POINTER(Buffer)]
    lib.pp_decompress.argtypes = [C.c_char_p, C.c_size_t, C.c_uint32, C.c_void_p, PROGRESS, C.c_void_p, C.POINTER(Buffer)]
    lib.pp_inspect.argtypes = [C.c_char_p, C.c_size_t, C.POINTER(Buffer)]
    no_progress = PROGRESS()

    def take(buf):
        out = C.string_at(buf.data, buf.len)
        lib.pp_buffer_free(C.byref(buf))
        return out

    def compress(data, opt):
        buf = Buffer()
        st = lib.pp_compress(data, len(data), C.byref(opt), None, no_progress, None, C.byref(buf))
        if st != 0:
            raise RuntimeError(lib.pp_last_error().decode())
        return take(buf)

    def decompress(data):
        buf = Buffer()
        st = lib.pp_decompress(data, len(data), 0, None, no_progress, None, C.byref(buf))
        if st != 0:
            raise RuntimeError(lib.pp_last_error().decode())
        return take(buf)

    failures = 0

    def check(ok, what):
        nonlocal failures
        print(("PASS " if ok else "FAIL ") + what)
        failures += not ok

    print("libpiedpiper", lib.pp_version().decode())
    manifest = json.load(open(os.path.join(GOLDEN, "manifest.json")))
    for case in manifest:
        data = open(os.path.join(GOLDEN, case["input"]), "rb").read()
        packed = open(os.path.join(GOLDEN, case["compressed"]), "rb").read()
        check(hashlib.sha256(decompress(packed)).hexdigest() == case["input_sha256"], f"{case['name']}: decompress golden")
        opt = Options()
        lib.pp_options_default(C.byref(opt))
        t = case["flags"].split()
        if not t or (len(t) == 1 and t[0][1:].isdigit()):  # defaults incl. built-in prior
            if t:
                opt.level = int(t[0][1:])
            check(hashlib.sha256(compress(data, opt)).hexdigest() == case["compressed_sha256"], f"{case['name']}: compress identical")
            continue
        f = dict(zip(t[::2], t[1::2]))
        opt.has_model = 1
        opt.model = Model(*(int(f[k]) for k in ("--d-model", "--layers", "--heads", "--d-ff", "--window", "--seg", "--streams")))
        tok = f.get("-T")  # absent: the default (auto)
        if tok is not None:
            opt.tokenizer, opt.bpe_vocab = (2, int(tok[4:])) if tok.startswith("bpe:") else (1, 4096)
        if any(k in f for k in ("--lr", "--beta1", "--clip")):
            lib.pp_preset_train(opt.level, C.byref(opt.train))
            opt.has_train = 1
            for k in ("lr", "beta1", "clip"):
                if "--" + k in f:
                    setattr(opt.train, k, float(f["--" + k]))
        check(hashlib.sha256(compress(data, opt)).hexdigest() == case["compressed_sha256"], f"{case['name']}: compress identical")

    text = b"C callers get the same bits as everyone else. " * 50
    opt = Options()
    lib.pp_options_default(C.byref(opt))
    opt.level = 1
    calls = []
    cb = PROGRESS(lambda d, t, u: calls.append(d) or 1)
    buf = Buffer()
    st = lib.pp_compress(text, len(text), C.byref(opt), None, cb, None, C.byref(buf))
    packed = take(buf)
    check(st == 0 and decompress(packed) == text and calls, "roundtrip with progress")
    stop = PROGRESS(lambda d, t, u: 0)
    buf = Buffer()
    check(lib.pp_compress(text, len(text), C.byref(opt), None, stop, None, C.byref(buf)) == 2, "cancellation")
    buf = Buffer()
    check(lib.pp_inspect(packed, len(packed), C.byref(buf)) == 0 and json.loads(take(buf))["original_size"] == len(text), "inspect")
    buf = Buffer()
    check(lib.pp_decompress(b"bogus", 5, 0, None, no_progress, None, C.byref(buf)) == 1 and b"magic" in lib.pp_last_error(), "error message")
    check(lib.pp_prior_load(b"not a prior", 11) is None and b"prior" in lib.pp_last_error(), "bad prior rejected")
    print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
