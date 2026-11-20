# libpiedpiper: C API

A shared and static library with a plain C ABI ([`include/piedpiper.h`](include/piedpiper.h))
for C, C++, Swift, Go (cgo), C#/.NET (P/Invoke), Zig, and anything else with a C FFI.
Output is bit-identical to every other PiedPiper build.

```bash
cargo build --release -p piedpiper-capi
# → target/release/libpiedpiper.{so,dylib,a} or piedpiper.{dll,lib}
```

```c
#include "piedpiper.h"

pp_options opt;
pp_options_default(&opt);
opt.level = 3;

pp_buffer out = {0};
if (pp_compress(data, len, &opt, NULL /*prior*/, NULL /*progress*/, NULL, &out) != PP_OK) {
    fprintf(stderr, "%s\n", pp_last_error());
} else {
    fwrite(out.data, 1, out.len, f);
    pp_buffer_free(&out);
}
```

- Progress: `int cb(uint64_t done, uint64_t total, void *user)` returns 0 to cancel, and then
  the call returns `PP_CANCELLED`. It may be called from a worker thread.
- Priors: `pp_prior_load(bytes, len)`, pass the result to compress/decompress, then free it
  with `pp_prior_free`.
- `pp_inspect` returns the header as JSON. Every `pp_buffer` is NUL-terminated.
- Errors: `pp_last_error()` is thread-local and valid until the next call.

`tests/capi_golden.py` exercises the whole API through `ctypes`, including the
cross-platform golden vectors:

```bash
python crates/capi/tests/capi_golden.py target/release/libpiedpiper.so
```
