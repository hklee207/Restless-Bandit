// Tests for the simulator core. No packages needed:  node test_sim.mjs
//
// Test 1: replay the Python env's random draws (trace.json) through sim.js and check the
//         reward of every round, for both policies, against the authors' env class.
// Test 2: 500 seeds with the JS random generator at N = 4, M = 1, checkpoint 2000, and
//         check the means against the replication's numbers.
// Any failure stops with exit code 1. Nothing here adjusts numbers to make them match.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
globalThis.window = globalThis;
require("./data.js");
const SIM = require("./sim.js");
const D = globalThis.SIM_DATA;
const trace = JSON.parse(readFileSync(path.join(here, "trace.json"), "utf8"));
const TOL = 1e-9;
const opt = { beta: D.meta.beta, c: D.meta.c, episodeLimit: D.meta.episodeLimit };

// ───────── Test 1: per-round rewards vs Python ─────────
console.log(`Test 1: replay ${trace.runs.length} Python runs (N=${trace.N}, M=${trace.M}, ${trace.steps} rounds each)`);
const policies = [
  ["index", D.closedForm, "closed-form index"],
  ["neurwin", SIM.tableFor(D, trace.checkpoint), `NeurWIN checkpoint ${trace.checkpoint}`],
];
let checked = 0, worst = 0;
for (const run of trace.runs) {
  for (const [key, table, label] of policies) {
    const ref = run[key];
    const st = new SIM.Station(table, run.spots, trace.M, opt);
    for (let t = 0; t < trace.steps; t++) {
      const before = st.spots.map((s) => [s.T, s.B]);
      const out = st.step();
      const err = Math.abs(out.reward - ref.rewards[t]);
      worst = Math.max(worst, err);
      checked++;
      const sameStates = JSON.stringify(before) === JSON.stringify(ref.states[t]);
      const sameChoice = JSON.stringify(out.chosen) === JSON.stringify(ref.chosen[t]);
      if (err > TOL || !sameStates || !sameChoice) {
        console.error(`\nDIVERGED: run ${run.run}, ${label}, round ${t}`);
        console.error(`  states (T,B) per spot   JS ${JSON.stringify(before)}   Python ${JSON.stringify(ref.states[t])}`);
        console.error(`  index values in JS      ${JSON.stringify(out.index)}`);
        console.error(`  spot(s) charged         JS ${JSON.stringify(out.chosen)}   Python ${JSON.stringify(ref.chosen[t])}`);
        console.error(`  reward this round       JS ${out.reward}   Python ${ref.rewards[t]}   |diff| ${err}`);
        process.exit(1);
      }
    }
    const dErr = Math.abs(st.discounted - ref.discountedTotal);
    console.log(`  run ${run.run}  ${label.padEnd(24)} discounted total ${st.discounted.toFixed(6)}  Python ${ref.discountedTotal.toFixed(6)}  |diff| ${dErr.toExponential(1)}`);
    if (dErr > TOL) { console.error("DIVERGED: discounted total differs"); process.exit(1); }
  }
}
console.log(`  PASS: ${checked} rounds match Python (states, choices, rewards). Largest reward difference ${worst.toExponential(1)} (limit ${TOL}).\n`);

// ───────── Test 2: 500 seeds with the JS generator ─────────
const seeds = Array.from({ length: 500 }, (_, i) => i + 1);
console.log(`Test 2: ${seeds.length} seeds, N=4, M=1, checkpoint 2000, JS random generator`);
const r = SIM.runPaired(4, 1, D.closedForm, SIM.tableFor(D, 2000), seeds, opt);
const s = SIM.pairedSummary(r.a, r.b);
const se = (x) => { const m = SIM.mean(x); return Math.sqrt(x.reduce((q, v) => q + (v - m) ** 2, 0) / (x.length - 1) / x.length); };
console.log(`  index mean     ${s.meanA.toFixed(2)}  (standard error ${se(r.a).toFixed(2)})   expected -105 +/- 3`);
console.log(`  NeurWIN mean   ${s.meanB.toFixed(2)}  (standard error ${se(r.b).toFixed(2)})   expected  -99 +/- 3`);
console.log(`  paired difference (NeurWIN - index)  ${s.diff >= 0 ? "+" : ""}${s.diff.toFixed(2)}  (standard error ${s.se.toFixed(2)})   expected +5.8 +/- 0.6`);
console.log(`  NeurWIN better in ${s.bBetter} of ${s.n} seeds`);
const checks = [
  ["index mean", s.meanA, -105, 3],
  ["NeurWIN mean", s.meanB, -99, 3],
  ["paired difference", s.diff, 5.8, 0.6],
];
let ok = true;
for (const [name, got, want, tol] of checks) {
  if (Math.abs(got - want) > tol) { ok = false; console.error(`  OUT OF RANGE: ${name} = ${got.toFixed(3)}, expected ${want} +/- ${tol}`); }
}
if (!ok) process.exit(1);
console.log("  PASS: all three inside the expected ranges.");
