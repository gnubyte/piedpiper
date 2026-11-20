//! Development aid: counts VP9 frames by coding mode (frame-parallel, error-resilient or
//! backward-adaptive) for every file given.
fn main() {
    for path in std::env::args().skip(1) {
        let Ok(data) = std::fs::read(&path) else { continue };
        let Some((_, samples)) = piedpiper_media::mkv::find_track(&data, b"V_VP9") else { continue };
        let (mut par, mut er, mut back, mut other) = (0, 0, 0, 0);
        for &(o, s) in &samples {
            let smp = &data[o as usize..o as usize + s as usize];
            for (fo, fl) in piedpiper_media::vp9::split_superframe(smp).0 {
                match piedpiper_media::vp9::frame_mode_flags(&smp[fo..fo + fl]) {
                    Some((_, true, _)) => er += 1,
                    Some((_, false, true)) => par += 1,
                    Some((_, false, false)) => back += 1,
                    None => other += 1,
                }
            }
        }
        println!("{:>10} B  parallel {par:>5}  error-resilient {er:>5}  BACKWARD {back:>5}  other {other:>4}  {path}", data.len());
    }
}
