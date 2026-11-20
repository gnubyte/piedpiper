"""PiedPiper — neural lossless compression.

Implements the TufaZip method: a small Transformer predicts the next token, an arithmetic
coder codes it with that distribution, and after every segment both encoder and decoder
take the same Adam step on what was just coded — so the model weights never need to be
stored. Output is bit-identical across Windows, Linux, macOS, Android, WebAssembly and
Python.

>>> import piedpiper
>>> packed = piedpiper.compress(b"hello hello hello hello", level=1)
>>> piedpiper.decompress(packed)
b'hello hello hello hello'

Compression is CPU-heavy (roughly 5-50 KB/s depending on level and input size); the GIL is
released while it runs, and Ctrl+C cancels.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from . import _native
from ._native import CancelledError, PiedPiperError, Prior

__version__: str = _native.__version__
FORMAT_VERSION: int = _native.FORMAT_VERSION
MIN_LEVEL: int = _native.MIN_LEVEL
MAX_LEVEL: int = _native.MAX_LEVEL
DEFAULT_LEVEL: int = _native.DEFAULT_LEVEL
MAGIC = b"PIED"
EXTENSION = ".pp"

BytesLike = Union[bytes, bytearray, memoryview]
PriorLike = Union[Prior, bytes, bytearray, memoryview, str, "os.PathLike[str]", None]
Progress = Optional[Callable[[int, int], Optional[bool]]]

__all__ = [
    "compress", "decompress", "info", "is_compressed", "compress_file", "decompress_file",
    "train_prior", "preset", "crc32", "Prior", "PiedPiperError", "CancelledError",
    "MIN_LEVEL", "MAX_LEVEL", "DEFAULT_LEVEL", "FORMAT_VERSION", "MAGIC", "EXTENSION",
]


def _bytes(data: Any, what: str = "data") -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        raise TypeError(f"{what} must be bytes-like, not str (encode it first)")
    try:
        return bytes(memoryview(data))
    except TypeError:
        raise TypeError(f"{what} must be bytes-like, not {type(data).__name__}") from None


def _prior(prior: PriorLike) -> Optional[Prior]:
    if prior is None or isinstance(prior, Prior):
        return prior
    if isinstance(prior, (str, os.PathLike)):
        return Prior.from_file(os.fspath(prior))
    return Prior(_bytes(prior, "prior"))


def compress(
    data: BytesLike,
    level: int = DEFAULT_LEVEL,
    *,
    tokenizer: str = "auto",
    bpe_vocab: int = 4096,
    threads: int = 0,
    prior: PriorLike = None,
    progress: Progress = None,
    allow_stored: bool = True,
    model: Optional[Dict[str, int]] = None,
    train: Optional[Dict[str, float]] = None,
    builtin_prior: bool = True,
) -> bytes:
    """Compress ``data``.

    :param level: 1 (fastest) to 9 (largest network). Levels 1-3 start text-like input from
        built-in priors (best for files up to a few MB); 4-9 suit large inputs.
    :param tokenizer: ``"auto"`` (word-level BPE for large text, bytes otherwise),
        ``"bytes"`` or ``"bpe"`` (vocabulary of up to ``bpe_vocab`` symbols).
    :param threads: worker threads, 0 = all cores. Never changes the output.
    :param prior: a :class:`Prior`, prior-file bytes, or a path. Needed again to decompress.
    :param progress: ``callback(done_bytes, total_bytes)``; return ``False`` to cancel
        (raises :class:`CancelledError`).
    :param model: advanced network override: ``d_model, n_layer, n_head, d_ff, window,
        seg_len, n_streams``.
    :param train: advanced training overrides on top of the level preset: ``lr, lr_min,
        warmup, decay, beta1, beta2, eps, clip, init_scale, seed``.
    :param builtin_prior: use the built-in text prior when no ``prior`` is given.
    """
    return _native.compress(
        _bytes(data), level, tokenizer, bpe_vocab, threads, _prior(prior), progress, allow_stored, model, train,
        builtin_prior,
    )


def decompress(
    data: BytesLike,
    *,
    threads: int = 0,
    prior: PriorLike = None,
    progress: Progress = None,
) -> bytes:
    """Decompress a PiedPiper stream (verifies its CRC-32)."""
    return _native.decompress(_bytes(data), threads, _prior(prior), progress)


def info(data: BytesLike) -> Dict[str, Any]:
    """Header of a compressed stream: sizes, tokenizer, network shape, training recipe."""
    return _native.info(_bytes(data))


def is_compressed(data: BytesLike) -> bool:
    """True if ``data`` starts with the PiedPiper magic bytes."""
    return bytes(memoryview(data)[:4]) == MAGIC


def crc32(data: BytesLike) -> int:
    return _native.crc32(_bytes(data))


def preset(level: int = DEFAULT_LEVEL) -> Tuple[Dict[str, int], Dict[str, float]]:
    """``(model, train)`` dicts of a compression level's preset."""
    return _native.preset(level)


def train_prior(
    corpus: BytesLike,
    *,
    level: int = DEFAULT_LEVEL,
    epochs: int = 1,
    tokenizer: str = "bytes",
    bpe_vocab: int = 4096,
    threads: int = 0,
    init: PriorLike = None,
    model: Optional[Dict[str, int]] = None,
    train: Optional[Dict[str, float]] = None,
    progress: Progress = None,
) -> Tuple[bytes, List[float]]:
    """Train a prior on ``corpus``. Returns ``(prior_file_bytes, bits_per_byte_per_epoch)``.

    Load the result with ``Prior(bytes)`` and pass it as ``prior=`` to compress/decompress.
    """
    return _native.train_prior(
        _bytes(corpus, "corpus"), level, epochs, tokenizer, bpe_vocab, threads, _prior(init), model, train, progress
    )


def _write_atomic(path: str, data: bytes, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(path)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def compress_file(src: Union[str, "os.PathLike[str]"], dst: Optional[str] = None, *, overwrite: bool = False, **kw: Any) -> str:
    """Compress ``src`` to ``dst`` (default ``src + '.pp'``); returns the output path."""
    src = os.fspath(src)
    dst = dst or src + EXTENSION
    with open(src, "rb") as f:
        data = f.read()
    _write_atomic(dst, compress(data, **kw), overwrite)
    return dst


def decompress_file(src: Union[str, "os.PathLike[str]"], dst: Optional[str] = None, *, overwrite: bool = False, **kw: Any) -> str:
    """Decompress ``src`` to ``dst`` (default: ``src`` without ``.pp``); returns the output path."""
    src = os.fspath(src)
    if dst is None:
        dst = src[: -len(EXTENSION)] if src.endswith(EXTENSION) else src + ".out"
    with open(src, "rb") as f:
        data = f.read()
    _write_atomic(dst, decompress(data, **kw), overwrite)
    return dst
