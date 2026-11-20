//! Development aid: how many VP9 frames and tiles of a file repeat byte for byte.
use piedpiper_media::vp9::{split_superframe, stream};
use std::collections::HashMap;

fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read");
        let Some((_, samples)) = piedpiper_media::mkv::find_track(&data, b"V_VP9") else { continue };
        let mut frames: HashMap<&[u8], usize> = HashMap::new();
        let (mut dup_frames, mut dup_frame_bytes, mut total) = (0, 0, 0);
        for &(o, s) in &samples {
            let smp = &data[o as usize..o as usize + s as usize];
            let (fr, _) = split_superframe(smp);
            for (fo, fl) in fr {
                let f = &smp[fo..fo + fl];
                total += fl;
                if frames.insert(f, 1).is_some() {
                    dup_frames += 1;
                    dup_frame_bytes += fl;
                }
            }
        }
        let tiles = stream::tile_repeats(&data);
        println!(
            "{path}: frames repeated {dup_frames} ({dup_frame_bytes} of {total} B); tiles repeated {} ({} B of {} B tile data)",
            tiles.0, tiles.1, tiles.2
        );
    }
}
