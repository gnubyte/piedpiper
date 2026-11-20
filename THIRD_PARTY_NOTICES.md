# Third-party notices

PiedPiper is MIT-licensed. Builds with the default `jpeg`, `deflate` and `media` features (the
CLI, the C library, the Python wheel, the npm package, the Android library and the Windows and
macOS desktop installers) also contain the following third-party code. Ship this file,
together with the license and notice files it points to, alongside binary distributions.

## lepton_jpeg 0.5.8

Lepton JPEG compression, Rust port. Copyright (c) Microsoft Corporation.
Licensed under the Apache License, Version 2.0: [third_party/lepton_jpeg/LICENSE.txt](third_party/lepton_jpeg/LICENSE.txt).
Notices: [third_party/lepton_jpeg/NOTICE.txt](third_party/lepton_jpeg/NOTICE.txt).
PiedPiper ships a modified copy. The changes are listed in
[third_party/lepton_jpeg/PIEDPIPER-PATCHES.md](third_party/lepton_jpeg/PIEDPIPER-PATCHES.md).

Per its NOTICE, lepton_jpeg itself includes parts of:

- **Dropbox Lepton** (<https://github.com/dropbox/lepton>), Apache License 2.0.
- **libwebp** (<https://github.com/webmproject/libwebp>), BSD 3-Clause License.
- **uncmpJPG**, BSD 2-Clause License.

## preflate-rs 0.7.6

Exact reconstruction of DEFLATE streams (used by the `deflate` feature, stream mode 4).
Copyright (c) Microsoft Corporation. Licensed under the Apache License, Version 2.0 (the
same license text as [third_party/lepton_jpeg/LICENSE.txt](third_party/lepton_jpeg/LICENSE.txt)).
Used unmodified from crates.io; the project publishes no NOTICE file.

preflate-rs depends on the `cabac` crate, which is licensed under LGPL-3.0-or-later.
PiedPiper does **not** include it: the workspace replaces it (`[patch.crates-io]` in the root
`Cargo.toml`) with [crates/cabac-shim](crates/cabac-shim), PiedPiper's own MIT-licensed
implementation of the few interfaces preflate-rs calls. Anyone building PiedPiper's crates
outside this workspace must apply the same patch to stay free of LGPL code.

## Tables from video codec reference implementations

The lossless video recoders (`crates/media`, the default `media` feature, stream mode 2) must use
exactly the probability and context tables each codec defines. The files that hold them name
their sources:

- **libvpx**, the WebM Project's VP8/VP9 library (<https://chromium.googlesource.com/webm/libvpx>,
  `main` branch as of November 2026): the VP8 and VP9 recoders in `crates/media/src/vp8/` and
  `crates/media/src/vp9/` follow libvpx's decoder and boolean coder closely, and their tables
  (`vp8/tables.rs`, `vp9/tables.rs`) were generated from its `vp8/common` and `vp9/common`
  sources. BSD 3-Clause License, with Google's additional patent grant for the WebM codecs; both
  are reproduced below.
- **OpenH264** (<https://github.com/cisco/openh264>): the tables in
  `crates/media/src/h264/tables.rs` were generated from the H.264 specification's tables as
  transcribed in OpenH264. BSD 2-Clause License, reproduced below.
- **HM**, the HEVC reference software (<https://vcgit.hhi.fraunhofer.de/jvet/HM>): the context
  tables in `crates/media/src/hevc/tables.rs` follow ITU-T H.265 and were checked value by value
  against HM's `TLibCommon/ContextTables.h`. BSD 3-Clause License, reproduced below.

### libvpx: license

```text
Copyright (c) 2010, The WebM Project authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * Neither the name of Google, nor the WebM Project, nor the names
    of its contributors may be used to endorse or promote products
    derived from this software without specific prior written
    permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### libvpx: additional IP rights grant (patents)

```text
Additional IP Rights Grant (Patents)
------------------------------------

"These implementations" means the copyrightable works that implement the WebM
codecs distributed by Google as part of the WebM Project.

Google hereby grants to you a perpetual, worldwide, non-exclusive, no-charge,
royalty-free, irrevocable (except as stated in this section) patent license to
make, have made, use, offer to sell, sell, import, transfer, and otherwise
run, modify and propagate the contents of these implementations of WebM, where
such license applies only to those patent claims, both currently owned by
Google and acquired in the future, licensable by Google that are necessarily
infringed by these implementations of WebM. This grant does not include claims
that would be infringed only as a consequence of further modification of these
implementations. If you or your agent or exclusive licensee institute or order
or agree to the institution of patent litigation or any other patent
enforcement activity against any entity (including a cross-claim or
counterclaim in a lawsuit) alleging that any of these implementations of WebM
or any code incorporated within any of these implementations of WebM
constitute direct or contributory patent infringement, or inducement of
patent infringement, then any patent rights granted to you under this License
for these implementations of WebM shall terminate as of the date such
litigation is filed.
```

### OpenH264: license

```text
Copyright (c) 2013, Cisco Systems
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### HM: license

```text
The copyright in this software is being made available under the BSD
License, included below. This software may be subject to other third party
and contributor rights, including patent rights, and no such rights are
granted under this license.

Copyright (c) 2010-2015, ITU/ISO/IEC
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
```

## Other Rust crates

Every other compiled-in dependency is available under a permissive license (MIT,
Apache-2.0, Zlib, BSD, Unlicense or Unicode-3.0, usually dual MIT/Apache-2.0). Their license
texts are collected in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), generated by
`python scripts/gen_licenses.py`; regenerate it whenever dependencies change and ship it with
binary distributions together with this file.

## Where the packages carry these files

Every package (the CLI and C library archives, the Python wheel, the npm package, the Android
library and the desktop installers) includes [LICENSE](LICENSE) (PiedPiper's MIT license),
this file, [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) and lepton_jpeg's `LICENSE.txt`
and `NOTICE.txt`:

| Package | Location |
|---|---|
| Python wheel | `piedpiper-<version>.dist-info/licenses/` (PEP 639 `license-files`), with lepton_jpeg's files named `LICENSE-lepton_jpeg.txt` and `NOTICE-lepton_jpeg.txt` |
| npm package | the package root (`package/` in the tarball), in this repository's layout |
| Android AAR | `META-INF/piedpiper/` in the AAR's `classes.jar` (and so in the APK of every app built with it), in this repository's layout |
| CLI and C library archives | the archive's top folder, with lepton_jpeg's files named `LICENSE-lepton_jpeg.txt` and `NOTICE-lepton_jpeg.txt` |
| Windows MSI | the install folder (`C:\Program Files\PiedPiper`), with lepton_jpeg's files named `LICENSE-lepton_jpeg.txt` and `NOTICE-lepton_jpeg.txt` |
| macOS DMG | `PiedPiper.app/Contents/Resources/Licenses/`, with the same names as the MSI |

maturin and npm only pack files from their own directories, so `bindings/python` and
`bindings/js` keep copies. Run `python bindings/sync_notices.py` after changing any of these files
or regenerating THIRD_PARTY_LICENSES.md (`bindings/js/build.py` refreshes the npm copies itself).
The Android build copies them from the repository root every time.

The desktop app ([apps/desktop](apps/desktop)) is a Cargo workspace of its own, which adds the
platform crates windows-rs and objc2 (MIT / Apache-2.0). Its installers therefore ship
`apps/desktop/THIRD_PARTY_LICENSES.md` in place of the root file; regenerate it with
`python scripts/gen_licenses.py --workspace apps/desktop --out apps/desktop/THIRD_PARTY_LICENSES.md`.
That workspace applies the same `cabac` patch as the root.
