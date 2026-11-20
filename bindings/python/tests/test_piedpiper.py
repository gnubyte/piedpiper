"""Tests for the Python bindings (run with `python -m pytest` or `python -m unittest`)."""
import hashlib
import json
import os
import random
import tempfile
import unittest

import piedpiper

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testdata", "golden")


def read(path):
    with open(path, "rb") as f:
        return f.read()


def flags_to_kwargs(flags):
    """Map the CLI flags recorded in testdata/golden/manifest.json to keyword arguments."""
    t = flags.split()
    model, train, kw = {}, {}, {}
    keys = {"--d-model": "d_model", "--layers": "n_layer", "--heads": "n_head", "--d-ff": "d_ff",
            "--window": "window", "--seg": "seg_len", "--streams": "n_streams"}
    i = 0
    while i < len(t):
        k = t[i]
        if len(k) == 2 and k[0] == "-" and k[1].isdigit():
            kw["level"] = int(k[1])
            i += 1
            continue
        v = t[i + 1]
        if k in keys:
            model[keys[k]] = int(v)
        elif k in ("--lr", "--beta1", "--clip"):
            train[k[2:]] = float(v)
        elif k == "-T":
            if v.startswith("bpe:"):
                kw["tokenizer"], kw["bpe_vocab"] = "bpe", int(v[4:])
            else:
                kw["tokenizer"] = v
        else:
            raise ValueError(k)
        i += 2
    if model:
        kw["model"] = model
    if train:
        kw["train"] = train
    return kw


class GoldenVectors(unittest.TestCase):
    """Files produced by the native Windows build must decode bit-exactly here, and
    re-compressing must produce the identical bytes."""

    def test_golden(self):
        manifest = json.loads(read(os.path.join(GOLDEN, "manifest.json")))
        self.assertGreaterEqual(len(manifest), 3)
        for case in manifest:
            with self.subTest(case=case["name"]):
                data = read(os.path.join(GOLDEN, case["input"]))
                packed = read(os.path.join(GOLDEN, case["compressed"]))
                out = piedpiper.decompress(packed)
                self.assertEqual(hashlib.sha256(out).hexdigest(), case["input_sha256"])
                again = piedpiper.compress(data, **flags_to_kwargs(case["flags"]))
                self.assertEqual(hashlib.sha256(again).hexdigest(), case["compressed_sha256"])
                self.assertEqual(piedpiper.info(packed)["original_size"], case["input_size"])


class Api(unittest.TestCase):
    TEXT = b"The piper plays; the model listens, predicts and learns. " * 60

    def test_roundtrip_and_types(self):
        for data in (b"", b"x", self.TEXT, bytearray(self.TEXT), memoryview(self.TEXT)):
            packed = piedpiper.compress(data, level=1)
            self.assertTrue(piedpiper.is_compressed(packed))
            self.assertEqual(piedpiper.decompress(packed), bytes(data))
        with self.assertRaises(TypeError):
            piedpiper.compress("not bytes")

    def test_compresses_text(self):
        packed = piedpiper.compress(self.TEXT, level=1)
        self.assertLess(len(packed), len(self.TEXT) // 2)
        meta = piedpiper.info(packed)
        self.assertEqual(meta["original_size"], len(self.TEXT))
        self.assertFalse(meta["stored"])
        self.assertIn("model", meta)

    def test_threads_do_not_change_output(self):
        a = piedpiper.compress(self.TEXT, level=2, threads=1)
        b = piedpiper.compress(self.TEXT, level=2, threads=4)
        self.assertEqual(a, b)

    def test_progress_and_cancel(self):
        seen = []
        piedpiper.compress(self.TEXT, level=1, progress=lambda d, t: seen.append((d, t)))
        self.assertTrue(seen)
        self.assertEqual(seen[-1][0], seen[-1][1])
        with self.assertRaises(piedpiper.CancelledError):
            piedpiper.compress(self.TEXT, level=1, progress=lambda d, t: False)

    def test_callback_exception_propagates(self):
        def boom(d, t):
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            piedpiper.compress(self.TEXT, level=1, progress=boom)

    def test_corruption(self):
        packed = bytearray(piedpiper.compress(self.TEXT, level=1))
        packed[len(packed) // 2 + 10] ^= 0x20
        with self.assertRaises(piedpiper.PiedPiperError):
            piedpiper.decompress(bytes(packed))
        with self.assertRaises(piedpiper.PiedPiperError):
            piedpiper.decompress(b"garbage")

    def test_bad_options(self):
        with self.assertRaises(piedpiper.PiedPiperError):
            piedpiper.compress(self.TEXT, level=42)
        with self.assertRaises(ValueError):
            piedpiper.compress(self.TEXT, tokenizer="nope")

    def test_prior(self):
        corpus = b"the cat sat on the mat. the dog sat on the log. " * 120
        model = {"d_model": 32, "n_layer": 1, "n_head": 2, "d_ff": 32, "window": 16, "seg_len": 8, "n_streams": 4}
        blob, bpb = piedpiper.train_prior(corpus, epochs=2, model=model)
        self.assertEqual(len(bpb), 2)
        prior = piedpiper.Prior(blob)
        self.assertGreater(prior.n_params, 0)
        self.assertEqual(prior.vocab_size, 256)
        self.assertEqual(prior.model["d_model"], 32)
        text = b"the dog sat on the mat. the cat sat on the log. " * 10
        with_p = piedpiper.compress(text, level=1, prior=prior)
        without = piedpiper.compress(text, level=1)
        self.assertLess(len(with_p), len(without))
        self.assertEqual(piedpiper.decompress(with_p, prior=blob), text)
        with self.assertRaises(piedpiper.PiedPiperError):
            piedpiper.decompress(with_p)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.ppp")
            with open(path, "wb") as f:
                f.write(blob)
            self.assertEqual(piedpiper.decompress(with_p, prior=path), text)

    def test_files(self):
        rnd = random.Random(1)
        data = bytes(rnd.choice(b"abcde \n") for _ in range(5000))
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "a.txt")
            with open(src, "wb") as f:
                f.write(data)
            out = piedpiper.compress_file(src, level=1)
            self.assertEqual(out, src + ".pp")
            os.remove(src)
            back = piedpiper.decompress_file(out)
            self.assertEqual(read(back), data)
            with self.assertRaises(FileExistsError):
                piedpiper.decompress_file(out)


if __name__ == "__main__":
    unittest.main()
