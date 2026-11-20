// Node test runner: `node test/golden.mjs` (after `python build.py`).
import { readFile } from 'node:fs/promises';
import { init } from '../src/index.js';
import { runGolden } from './golden-lib.js';

const dir = new URL('../../../testdata/golden/', import.meta.url);
await init();
const { failed } = await runGolden((name) => readFile(new URL(name, dir)), (m) => console.log(m));
process.exit(failed ? 1 : 0);
