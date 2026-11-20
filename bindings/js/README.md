# piedpiper (JavaScript / WebAssembly)

Neural lossless compression in the browser, Node.js, Deno and Bun, using the
[TufaZip](https://tufalabs.ai/research/tufazip/) method. An online-learning Transformer
drives an arithmetic coder, and the decoder re-learns the weights as it decodes, so they
are never stored.

The codec is the Rust core compiled to WebAssembly (with SIMD128) and has no dependencies.
The module is about 3 MB, almost all of it the three built-in text priors, which streams
from every PiedPiper build may reference. It is **bit-identical** to the native CLI, Python and Android builds, so files
move freely between them.

```js
import { init, compress, decompress, info } from 'piedpiper';

await init();                                  // loads piedpiper.wasm next to the module
const packed = compress(bytes, { level: 3 });  // Uint8Array | ArrayBuffer | string
const plain = decompress(packed);
info(packed);  // { originalSize, compressedSize, tokenizer, model: {...}, train: {...}, ... }
```

Options: `level` (1–9), `tokenizer` (`'auto' | 'bytes' | 'bpe'`), `bpeVocab`, `prior`
(prior file bytes), `onProgress(done, total)` (return `false` to cancel), `allowStored`,
and the advanced `model` / `train` overrides (see `src/index.d.ts`).

## Off the main thread

Compression runs at tens of KB/s. In the browser, run it in a worker:

```js
import { createWorker } from 'piedpiper';
const pp = createWorker();
const packed = await pp.compress(file, { level: 3, onProgress: (d, t) => bar.value = d / t });
pp.terminate();   // also the way to cancel
```

## Loading the module elsewhere

`init()` accepts a URL, the wasm bytes, or a compiled `WebAssembly.Module`. Bundlers can
import `piedpiper/piedpiper.wasm?url` (Vite) and pass the result. `initSync(bytes)` is
available for workers and Node.

## Building and testing

```bash
python build.py                  # cargo build --target wasm32-unknown-unknown + copy
node test/golden.mjs             # golden vectors made by the native build must match
```

To run the tests in a browser, serve the repository root (`python -m http.server`) and open
`/bindings/js/test/golden.html`. `demo/index.html` is a drag-and-drop compressor.
