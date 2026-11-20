//! Development tool: recodes every AC-3 frame found in a file (any container), checks the
//! round trip and prints the sizes. `cargo run --release --example ac3rt -- FILE...`
use piedpiper_media::ac3;
use std::time::Instant;

fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read");
        let mut frames = Vec::new();
        ac3::scan(&data, 0, data.len(), &mut frames);
        let mut probe = ac3::Probe::default();
        let ok: Vec<(usize, usize)> = frames.iter().copied().filter(|&(o, s)| ac3::parses(&data[o..o + s], &mut probe)).collect();
        let total: usize = ok.iter().map(|&(_, s)| s).sum();
        // runs of about 4 MB, coded in parallel
        let mut runs: Vec<Vec<(usize, usize)>> = vec![Vec::new()];
        let mut acc = 0;
        for &f in &ok {
            let run_mb: usize = std::env::var("AC3_RUN_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
            if acc >= run_mb << 20 {
                runs.push(Vec::new());
                acc = 0;
            }
            runs.last_mut().unwrap().push(f);
            acc += f.1;
        }
        if std::env::var_os("AC3_COSTS").is_some() {
            let first: Vec<(usize, usize)> = runs[0].clone();
            let tot: f64 = ac3::cost_breakdown(&data, &first, 22).iter().map(|x| x.2).sum();
            for (name, n, c) in ac3::cost_breakdown(&data, &first, 22) {
                if n > 0 {
                    println!("  {name:<14} bits {n:>9} ({:>5.1}%)  model {:>9.0} B ({:>5.1}% of out)  saved {:>5.1}%", 100.0 * n as f64 / (first.iter().map(|f| f.1).sum::<usize>() * 8) as f64, c / 8.0, 100.0 * c / tot, 100.0 * (1.0 - c / n as f64));
                }
            }
        }
        let t0 = Instant::now();
        let coded: Vec<Vec<u8>> = std::thread::scope(|sc| {
            let hs: Vec<_> = runs.iter().map(|r| sc.spawn(|| ac3::encode_run(&data, r, 22).expect("encode"))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let tc = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let good = std::thread::scope(|sc| {
            let hs: Vec<_> = runs
                .iter()
                .zip(&coded)
                .map(|(r, c)| {
                    let data = &data;
                    sc.spawn(move || {
                        let dec = ac3::decode_run(c, 22, data.len()).expect("decode");
                        dec.len() == r.len() && dec.iter().zip(r).all(|((o, f), &(o2, s2))| *o == o2 && f[..] == data[o2..o2 + s2])
                    })
                })
                .collect();
            hs.into_iter().all(|h| h.join().unwrap())
        });
        let td = t1.elapsed().as_secs_f64();
        let out: usize = coded.iter().map(Vec::len).sum();
        println!(
            "{path}: {} AC-3 frames found, {} parsed ({} B) -> {} B ({:.2}% smaller) comp {tc:.2}s decomp {td:.2}s {}",
            frames.len(),
            ok.len(),
            total,
            out,
            100.0 * (1.0 - out as f64 / total.max(1) as f64),
            if good { "OK" } else { "MISMATCH" }
        );
    }
}
