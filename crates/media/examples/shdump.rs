//! Prints slice-header facts of an H.264 MP4 (development aid).
use piedpiper_media::{bits::unescape, h264::ParamSets, mp4};

fn main() {
    let path = std::env::args().nth(1).expect("file");
    let limit: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let data = std::fs::read(&path).expect("read");
    let t = mp4::find_avc_track(&data).expect("avc track");
    let mut ps = ParamSets::default();
    for n in &t.param_nals {
        ps.add_nal(n);
    }
    let mut shown = 0;
    let mut counts = std::collections::BTreeMap::new();
    for &(o, s) in &t.samples {
        let smp = &data[o as usize..(o + s as u64) as usize];
        let mut p = 0;
        while p + t.nal_len_size <= smp.len() {
            let mut l = 0usize;
            for k in 0..t.nal_len_size {
                l = l << 8 | smp[p + k] as usize;
            }
            p += t.nal_len_size;
            let nal = &smp[p..p + l];
            p += l;
            let ty = (nal[0] & 31) as u32;
            if ty == 7 || ty == 8 {
                ps.add_nal(nal);
                continue;
            }
            if ty != 1 && ty != 5 {
                continue;
            }
            let rbsp = unescape(&nal[1..]);
            let nri = ((nal[0] >> 5) & 3) as u32;
            match ps.slice_header(&rbsp, ty, nri) {
                Ok((sh, sps, pps)) => {
                    let key = format!("type {} ref {} spatial {} l0 {} l1 {}", sh.slice_type, nri, sh.direct_spatial, sh.num_ref_l0, sh.num_ref_l1);
                    *counts.entry(key).or_insert(0) += 1;
                    if shown < limit {
                        shown += 1;
                        println!(
                            "first_mb {} type {} nri {} frame_num {} idr {} qp {} spatial {} l0 {} l1 {} | poc_type {} prof {} t8 {} wp {} bipred {} {}x{}",
                            sh.first_mb, sh.slice_type, nri, sh.frame_num, sh.idr, sh.slice_qp, sh.direct_spatial, sh.num_ref_l0, sh.num_ref_l1,
                            sps.poc_type, sps.profile_idc, pps.transform_8x8_mode, pps.weighted_pred, pps.weighted_bipred_idc, sps.width_mbs, sps.height_mbs()
                        );
                    }
                }
                Err(_) => *counts.entry("unparsed".to_string()).or_insert(0) += 1,
            }
        }
    }
    for (k, v) in counts {
        println!("{v:>7}  {k}");
    }
}
