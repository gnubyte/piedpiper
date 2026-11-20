# piedpiper (Python)

Neural lossless compression for Python, using the
[TufaZip](https://tufalabs.ai/research/tufazip/) method. A small Transformer predicts the
next token and an arithmetic coder codes it with that distribution. After every segment,
encoder and decoder both take the same Adam step on what was just coded, so the model
weights are never stored. The decoder re-learns them.

The codec is a Rust core (bundled as a native extension). It is **bit-identical on every
platform**: a file compressed in Python decompresses with the CLI on Windows, Linux or macOS,
in the browser (WebAssembly), or on Android, and vice versa.

```python
import piedpiper

data = open("book.txt", "rb").read()
packed = piedpiper.compress(data, level=3)            # level 1 (fast) … 9 (strong)
assert piedpiper.decompress(packed) == data

piedpiper.info(packed)       # sizes, tokenizer, network shape, training recipe
piedpiper.compress_file("book.txt")                 # -> book.txt.pp
piedpiper.decompress_file("book.txt.pp", overwrite=True)
```

Neural compression is slow by design (roughly 5–50 KB/s, depending on level and input
size). The GIL is released while the codec runs. Progress and cancellation work like this:

```python
def progress(done, total):
    print(f"{done / total:.0%}")
    return True            # return False to cancel (raises piedpiper.CancelledError)

piedpiper.compress(data, progress=progress, threads=4)   # Ctrl+C also cancels
```

`threads` never changes the output bytes.

## Priors

TufaZip v2 starts online learning from an offline-trained prior. That helps most on small
inputs, where a model trained from scratch sees too little data to learn much:

```python
blob, bits_per_byte = piedpiper.train_prior(corpus, level=3, epochs=2)
prior = piedpiper.Prior(blob)                        # or Prior.from_file("text.ppp")
packed = piedpiper.compress(data, prior=prior)
piedpiper.decompress(packed, prior=prior)            # the same prior is required
```

## Building from source

```bash
pip install maturin
maturin build --release        # wheel in target/wheels (or -o dist)
pip install dist/piedpiper-*.whl
python -m unittest discover -s tests
```

The command line is `python -m piedpiper [-1..-9] [-d] FILE...`.
