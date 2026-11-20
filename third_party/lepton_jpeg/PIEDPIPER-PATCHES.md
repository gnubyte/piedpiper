# lepton_jpeg 0.5.8, vendored for PiedPiper

Source: the `lepton_jpeg` 0.5.8 crate from crates.io
(<https://github.com/microsoft/lepton_jpeg_rust>, Copyright (c) Microsoft Corporation,
Apache License 2.0). `LICENSE.txt` and `NOTICE.txt` come from the same project (the 0.5.8
crate package omits them).

PiedPiper uses it for stream mode 2 (lossless JPEG recompression). It is vendored because
the published crate cannot be used unmodified:

1. **WebAssembly.** `metrics.rs` and `structs/lepton_file_writer.rs` call
   `std::time::Instant::now()` in every encode/decode worker. That panics on
   `wasm32-unknown-unknown` (no clock). The timers are no-ops on that target now; they only
   feed log messages and metrics, never the output.
2. **Reproducible output.** `structs/lepton_file_writer.rs` wrote `git describe` of the
   directory the crate was built in into every stream header (via the `git-version` macro),
   so identical input could compress to different bytes depending on the build machine.
   The revision is now the constant `"0"`, which is also what a normal crates.io build
   produces. The `git-version` dependency is gone.
3. **Trimmed.** The development-only `micro_benchmark` module and feature, the test-data
   text files and the dev-dependencies were removed.

The Lepton stream format itself is unchanged: streams decode with upstream lepton_jpeg and
vice versa.
