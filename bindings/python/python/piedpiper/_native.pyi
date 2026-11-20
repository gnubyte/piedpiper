from typing import Any, Callable, Dict, List, Optional, Tuple

__version__: str
FORMAT_VERSION: int
MIN_LEVEL: int
MAX_LEVEL: int
DEFAULT_LEVEL: int

class PiedPiperError(Exception): ...
class CancelledError(PiedPiperError): ...

class Prior:
    def __init__(self, data: bytes) -> None: ...
    @staticmethod
    def from_file(path: str) -> "Prior": ...
    @property
    def id(self) -> int: ...
    @property
    def n_params(self) -> int: ...
    @property
    def tokenizer(self) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def file_size(self) -> int: ...
    @property
    def model(self) -> Dict[str, int]: ...

_Progress = Optional[Callable[[int, int], Optional[bool]]]

def compress(
    data: bytes,
    level: int = ...,
    tokenizer: str = ...,
    bpe_vocab: int = ...,
    threads: int = ...,
    prior: Optional[Prior] = ...,
    progress: _Progress = ...,
    allow_stored: bool = ...,
    model: Optional[Dict[str, int]] = ...,
    train: Optional[Dict[str, float]] = ...,
    builtin_prior: bool = ...,
) -> bytes: ...
def decompress(data: bytes, threads: int = ..., prior: Optional[Prior] = ..., progress: _Progress = ...) -> bytes: ...
def info(data: bytes) -> Dict[str, Any]: ...
def train_prior(
    corpus: bytes,
    level: int = ...,
    epochs: int = ...,
    tokenizer: str = ...,
    bpe_vocab: int = ...,
    threads: int = ...,
    init: Optional[Prior] = ...,
    model: Optional[Dict[str, int]] = ...,
    train: Optional[Dict[str, float]] = ...,
    progress: _Progress = ...,
) -> Tuple[bytes, List[float]]: ...
def preset(level: int) -> Tuple[Dict[str, int], Dict[str, float]]: ...
def crc32(data: bytes) -> int: ...
