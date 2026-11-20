//! Branch-call-jump (BCJ) filters for machine code, applied before neural coding.
//!
//! Executables encode call/jump targets as offsets relative to the instruction, so calls to
//! the same function look different everywhere. Converting them to absolute addresses
//! makes them repeat, which any predictor exploits. These are ports of the public-domain
//! filters in Igor Pavlov's LZMA SDK (x86 `E8/E9`) plus the AArch64 `BL` variant used by
//! xz. Both directions are exact inverses, so the transform is lossless by construction;
//! the codec additionally verifies the round trip before using it. See [`crate::filter`].

/// Instruction set whose branch encoding a filter rewrites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    X86,
    Arm64,
}

/// The instruction set of an executable (PE, ELF or Mach-O header), if `data` is one.
pub fn detect(data: &[u8]) -> Option<Arch> {
    if data.len() < 64 {
        return None;
    }
    // PE: "MZ" ... "PE\0\0" at e_lfanew, machine field after it.
    if data.starts_with(b"MZ") {
        let off = u32::from_le_bytes([data[0x3c], data[0x3d], data[0x3e], data[0x3f]]) as usize;
        if off + 6 <= data.len() && &data[off..off + 4] == b"PE\0\0" {
            return match u16::from_le_bytes([data[off + 4], data[off + 5]]) {
                0x014c | 0x8664 => Some(Arch::X86),
                0xaa64 => Some(Arch::Arm64),
                _ => None,
            };
        }
        return None;
    }
    // ELF: e_machine at 18 (little-endian files only).
    if data.starts_with(b"\x7fELF") && data[5] == 1 {
        return match u16::from_le_bytes([data[18], data[19]]) {
            3 | 62 => Some(Arch::X86),
            183 => Some(Arch::Arm64),
            _ => None,
        };
    }
    // Mach-O 64-bit little-endian: cputype at 4.
    if data.starts_with(&[0xcf, 0xfa, 0xed, 0xfe]) {
        return match u32::from_le_bytes([data[4], data[5], data[6], data[7]]) {
            0x0100_0007 => Some(Arch::X86),
            0x0100_000c => Some(Arch::Arm64),
            _ => None,
        };
    }
    None
}

/// Makes branch targets absolute (in place).
pub fn encode(arch: Arch, data: &mut [u8]) {
    match arch {
        Arch::X86 => x86_convert(data, true),
        Arch::Arm64 => arm64_convert(data, true),
    }
}

/// Inverts [`encode`] (in place).
pub fn decode(arch: Arch, data: &mut [u8]) {
    match arch {
        Arch::X86 => x86_convert(data, false),
        Arch::Arm64 => arm64_convert(data, false),
    }
}

#[inline]
fn test_ms_byte(b: u8) -> bool {
    b == 0 || b == 0xFF
}

/// x86 `E8`/`E9` filter (LZMA SDK `x86_Convert`, start address 0, whole buffer).
fn x86_convert(data: &mut [u8], encoding: bool) {
    const ALLOWED: [bool; 8] = [true, true, true, false, true, false, false, false];
    const BIT: [u32; 8] = [0, 1, 2, 2, 3, 3, 3, 3];
    let size = data.len();
    if size < 5 {
        return;
    }
    let ip: u32 = 5;
    let mut pos: usize = 0;
    let mut prev_pos: usize = usize::MAX; // "(SizeT)0 - 1"
    let mut prev_mask: u32 = 0;
    let limit = size - 4;
    loop {
        while pos < limit && data[pos] & 0xFE != 0xE8 {
            pos += 1;
        }
        if pos >= limit {
            break;
        }
        let dist = pos.wrapping_sub(prev_pos);
        if dist > 3 {
            prev_mask = 0;
        } else {
            prev_mask = (prev_mask << (dist as u32 - 1)) & 7;
            if prev_mask != 0 {
                let b = data[pos + 4 - BIT[prev_mask as usize] as usize];
                if !ALLOWED[prev_mask as usize] || test_ms_byte(b) {
                    prev_pos = pos;
                    prev_mask = ((prev_mask << 1) & 7) | 1;
                    pos += 1;
                    continue;
                }
            }
        }
        prev_pos = pos;
        if test_ms_byte(data[pos + 4]) {
            let mut src = u32::from_le_bytes([data[pos + 1], data[pos + 2], data[pos + 3], data[pos + 4]]);
            let cur = ip.wrapping_add(pos as u32);
            let mut dest;
            loop {
                dest = if encoding { cur.wrapping_add(src) } else { src.wrapping_sub(cur) };
                if prev_mask == 0 {
                    break;
                }
                let index = BIT[prev_mask as usize] * 8;
                let b = (dest >> (24 - index)) as u8;
                if !test_ms_byte(b) {
                    break;
                }
                src = dest ^ ((1u32 << (32 - index)) - 1);
            }
            data[pos + 4] = !(((dest >> 24) & 1).wrapping_sub(1)) as u8;
            data[pos + 3] = (dest >> 16) as u8;
            data[pos + 2] = (dest >> 8) as u8;
            data[pos + 1] = dest as u8;
            pos += 5;
        } else {
            prev_mask = ((prev_mask << 1) & 7) | 1;
            pos += 1;
        }
    }
}

/// AArch64 `BL imm26` filter (xz's ARM64 BCJ, BL part): 4-byte aligned little-endian words.
fn arm64_convert(data: &mut [u8], encoding: bool) {
    let mut i = 0;
    while i + 4 <= data.len() {
        let w = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        if w >> 26 == 0x25 {
            let pc = (i as u32) >> 2;
            let imm = w & 0x03FF_FFFF;
            let v = if encoding { imm.wrapping_add(pc) } else { imm.wrapping_sub(pc) } & 0x03FF_FFFF;
            data[i..i + 4].copy_from_slice(&(0x9400_0000 | v).to_le_bytes());
        }
        i += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn filters_are_exact_inverses() {
        let mut rng = Rng::new(17);
        for len in [0usize, 1, 4, 5, 9, 100, 4096, 70_001] {
            // Random bytes with plenty of E8/E9 and BL patterns.
            let mut data: Vec<u8> = (0..len)
                .map(|i| match rng.next_u64() % 8 {
                    0 => 0xE8,
                    1 => 0xE9,
                    2 if i % 4 == 3 => 0x94,
                    3 => 0x00,
                    4 => 0xFF,
                    _ => rng.next_u64() as u8,
                })
                .collect();
            let orig = data.clone();
            for f in [Arch::X86, Arch::Arm64] {
                encode(f, &mut data);
                decode(f, &mut data);
                assert_eq!(data, orig, "{f:?} len {len}");
            }
        }
    }

    #[test]
    fn detects_executables() {
        let mut pe = vec![0u8; 256];
        pe[..2].copy_from_slice(b"MZ");
        pe[0x3c] = 0x80;
        pe[0x80..0x84].copy_from_slice(b"PE\0\0");
        pe[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());
        assert_eq!(detect(&pe), Some(Arch::X86));
        pe[0x84..0x86].copy_from_slice(&0xaa64u16.to_le_bytes());
        assert_eq!(detect(&pe), Some(Arch::Arm64));
        let mut elf = vec![0u8; 64];
        elf[..4].copy_from_slice(b"\x7fELF");
        elf[5] = 1;
        elf[18] = 62;
        assert_eq!(detect(&elf), Some(Arch::X86));
        assert_eq!(detect(b"plain text, not an executable at all, really not one ever, no sir"), None);
    }
}
