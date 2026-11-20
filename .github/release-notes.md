## Download PiedPiper {{VERSION}}

| Your computer | Download | Then |
|---|---|---|
| **Windows 10 or 11** | **[PiedPiper-Windows-x64.msi](https://github.com/gnubyte/piedpiper/releases/download/v{{VERSION}}/PiedPiper-Windows-x64.msi)** | Run it and follow the installer. |
| **Mac** (macOS 11 or later, Apple Silicon or Intel) | **[PiedPiper-macOS.dmg](https://github.com/gnubyte/piedpiper/releases/download/v{{VERSION}}/PiedPiper-macOS.dmg)** | Open it, drag **PiedPiper** into **Applications**, and open it once from there. |

Then right-click files and folders:

- **Compress with PiedPiper**: everything you selected, any number of items, goes into one `.pp` archive next to it.
- **Extract here with PiedPiper** and **Extract to folder with PiedPiper** on `.pp` archives. Double-clicking an archive extracts it too.

On Windows 11 these commands are under **Show more options** (or press Shift+F10). On a Mac, Control-click and look under **Quick Actions** or **Services**. To pick the compression level, open PiedPiper from the Start menu (Windows) or from Applications (Mac).

These builds aren't code-signed yet, so the first launch asks you to confirm. On Windows, SmartScreen shows **More info**, then **Run anyway**. On a Mac, if PiedPiper won't open, go to **System Settings → Privacy & Security** and click **Open Anyway**.

### For developers

| File | What it is |
|---|---|
| `piedpiper-v{{VERSION}}-<platform>.zip` / `.tar.gz` | the `piedpiper` command-line tool and the C library, for Windows, Linux and macOS (x86-64 and ARM64) |
| `piedpiper-{{VERSION}}-*.whl` | Python wheels: `pip install <file>` |
| `piedpiper-{{VERSION}}.tgz` | npm package (WebAssembly): `npm install <file>` |
| `piedpiper-release.aar`, `app-release.apk` | Android library and demo app |

Files compressed by any of them open in all the others, on every platform.
