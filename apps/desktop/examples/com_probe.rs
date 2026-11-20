//! Test harness: asks COM for a class as a local server and prints the result.
//!
//!     cargo run --example com_probe -- {27A4092B-ED3D-4569-9C13-DF2ED29481D9}

#[cfg(windows)]
fn main() {
    use windows::core::{IUnknown, GUID};
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CLSCTX_LOCAL_SERVER, COINIT_APARTMENTTHREADED,
    };
    let arg = std::env::args().nth(1).expect("usage: com_probe {CLSID}");
    let clsid = GUID::try_from(arg.trim_matches(|c| c == '{' || c == '}')).expect("CLSID");
    unsafe {
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
        let r: windows::core::Result<IUnknown> = CoCreateInstance(&clsid, None, CLSCTX_LOCAL_SERVER);
        match r {
            Ok(_) => println!("created {arg}"),
            Err(e) => println!("{arg}: {:#010x} {}", e.code().0, e.message()),
        }
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("com_probe is a Windows test harness");
}
