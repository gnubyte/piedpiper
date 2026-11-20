// Shared golden-vector test (browser + Node). Every `.bin.pp` in testdata/golden was produced
// by the native Windows build; decompressing it here proves the WebAssembly build performs
// bit-identical float arithmetic, and re-compressing must give identical bytes.
import { compress, decompress, info, version, isCompressed, PiedPiperError, CancelledError } from '../src/index.js';

/** Maps the CLI flags recorded in manifest.json to JS options. */
export function flagsToOptions(flags) {
  const t = flags.trim().split(/\s+/).filter(Boolean);
  const model = {};
  const train = {};
  const opts = {};
  for (let i = 0; i < t.length; i++) {
    const k = t[i];
    const v = t[i + 1];
    const num = () => { i++; return Number(v); };
    switch (k) {
      case '--d-model': model.dModel = num(); break;
      case '--layers': model.layers = num(); break;
      case '--heads': model.heads = num(); break;
      case '--d-ff': model.dFf = num(); break;
      case '--window': model.window = num(); break;
      case '--seg': model.segLen = num(); break;
      case '--streams': model.streams = num(); break;
      case '--lr': train.lr = num(); break;
      case '--beta1': train.beta1 = num(); break;
      case '--clip': train.clip = num(); break;
      case '-T': {
        i++;
        if (v.startsWith('bpe:')) { opts.tokenizer = 'bpe'; opts.bpeVocab = Number(v.slice(4)); }
        else opts.tokenizer = v;
        break;
      }
      default:
        if (/^-[1-9]$/.test(k)) { opts.level = Number(k[1]); break; }
        throw new Error(`unhandled flag ${k}`);
    }
  }
  if (Object.keys(model).length) opts.model = model;
  if (Object.keys(train).length) opts.train = train;
  return opts;
}

async function sha256(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, '0')).join('');
}

const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

/** `load(name)` returns the bytes of testdata/golden/<name>. Returns {passed, failed}. */
export async function runGolden(load, log) {
  let passed = 0;
  let failed = 0;
  const check = (ok, what) => {
    if (ok) passed++;
    else failed++;
    log(`${ok ? 'PASS' : 'FAIL'} ${what}`);
  };
  log(`piedpiper ${version()} (wasm)`);
  const manifest = JSON.parse(new TextDecoder().decode(await load('manifest.json')));
  for (const c of manifest) {
    const input = await load(c.input);
    const packed = await load(c.compressed);
    check(isCompressed(packed), `${c.name}: magic`);
    const t0 = performance.now();
    const out = decompress(packed);
    const t1 = performance.now();
    check((await sha256(out)) === c.input_sha256 && eq(out, input), `${c.name}: decompress golden (${(t1 - t0).toFixed(0)} ms)`);
    const again = compress(input, flagsToOptions(c.flags));
    const t2 = performance.now();
    check((await sha256(again)) === c.compressed_sha256, `${c.name}: compress -> identical bytes (${(t2 - t1).toFixed(0)} ms)`);
    check(info(packed).originalSize === c.input_size, `${c.name}: info`);
  }
  // Roundtrip, progress, cancellation and errors.
  const text = new TextEncoder().encode('Pied Piper: middle-out? No — online-learning transformers. '.repeat(40));
  let calls = 0;
  const p = compress(text, { level: 1, onProgress: () => { calls++; } });
  check(eq(decompress(p), text) && calls > 0, `roundtrip level 1 with progress (${text.length} -> ${p.length})`);
  let cancelled = false;
  try { compress(text, { level: 1, onProgress: () => false }); } catch (e) { cancelled = e instanceof CancelledError; }
  check(cancelled, 'cancellation');
  let corrupt = false;
  const bad = p.slice();
  bad[(bad.length >> 1) + 10] ^= 0x10;
  try { decompress(bad); } catch (e) { corrupt = e instanceof PiedPiperError; }
  check(corrupt, 'corruption detected');
  check(eq(decompress(compress(new Uint8Array(0))), new Uint8Array(0)), 'empty input');
  log(`${failed === 0 ? 'ALL PASSED' : 'FAILURES'}: ${passed} passed, ${failed} failed`);
  return { passed, failed };
}
