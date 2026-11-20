// Module Web Worker used by `createWorker()`. Runs the synchronous codec off the main thread.
import { init, compress, decompress, info, CancelledError } from './index.js';

self.onmessage = async (e) => {
  const { id, op, data, opts = {} } = e.data;
  try {
    await init(opts.wasmUrl);
    let last = 0;
    const o = { ...opts };
    delete o.progress;
    delete o.wasmUrl;
    if (opts.progress) {
      o.onProgress = (done, total) => {
        const now = Date.now();
        if (now - last >= 100 || done === total) {
          last = now;
          self.postMessage({ id, progress: [done, total] });
        }
        return true;
      };
    }
    let result;
    if (op === 'compress') result = compress(data, o);
    else if (op === 'decompress') result = decompress(data, o);
    else if (op === 'info') result = info(data);
    else throw new Error(`unknown op ${op}`);
    if (result instanceof Uint8Array) self.postMessage({ id, result }, [result.buffer]);
    else self.postMessage({ id, result });
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message ? err.message : err), cancelled: err instanceof CancelledError });
  }
};
