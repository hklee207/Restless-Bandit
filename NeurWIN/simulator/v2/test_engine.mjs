// Checks engine.js against the authors' Python environment (trace from export_data.py).
import fs from 'fs';
const dir = process.argv[2];
eval(fs.readFileSync(dir + '/engine.js', 'utf8'));
const EV = globalThis.EV;
const data = JSON.parse(fs.readFileSync(dir + '/data.json', 'utf8'));
const trace = JSON.parse(fs.readFileSync(dir + '/data_trace.json', 'utf8'));

let worst = 0, pickMismatch = 0;
const tabCF = []; for (let T = 0; T < 13; T++) { tabCF.push([]); for (let B = 0; B < 10; B++) tabCF[T].push(EV.closedForm(T, B)); }
for (let T = 0; T < 13; T++) for (let B = 0; B < 10; B++) worst = Math.max(worst, Math.abs(tabCF[T][B] - data.closed[T][B]));
console.log('closed-form table max diff vs python:', worst.toExponential(2));

for (const [i, run] of trace.entries()) {
  for (const [name, score] of [['closed', EV.closedForm], ['neurwin', EV.tableScore(data.tables['2000'])]]) {
    const out = EV.runEpisode(run.draws, score, 1);
    let d = 0; out.rewards.forEach((r, t) => { d = Math.max(d, Math.abs(r - run[name].rewards[t])); });
    out.picks.forEach((p, t) => { if (p[0] !== run[name].picks[t]) pickMismatch++; });
    worst = Math.max(worst, d);
    const disc = run[name].rewards.reduce((s, r, t) => s + Math.pow(0.99, t) * r, 0);
    console.log(`run ${i} ${name}: max reward diff ${d.toExponential(2)}, total js ${out.disc.toFixed(4)} py ${disc.toFixed(4)}`);
  }
}
console.log('pick mismatches:', pickMismatch);
if (worst > 1e-9 || pickMismatch) { console.log('FAIL'); process.exit(1); }

const t0 = Date.now();
const c = EV.compare(4, 1, EV.closedForm, EV.tableScore(data.tables['2000']), 500, 1);
console.log(`500 runs N=4 M=1: index ${c.meanA.toFixed(2)}, NeurWIN ${c.meanB.toFixed(2)}, diff ${c.diff.toFixed(2)} ± ${c.se.toFixed(2)}, NeurWIN better in ${c.wins}/500 (${Date.now() - t0} ms)`);
for (const [N, M] of [[8, 2], [20, 5], [100, 25]]) {
  const r = EV.compare(N, M, EV.closedForm, EV.tableScore(data.tables['2000']), 50, 1);
  console.log(`50 runs N=${N} M=${M}: diff ${r.diff.toFixed(2)} (${(100 * r.diff / Math.abs(r.meanA)).toFixed(1)}%), better ${r.wins}/50`);
}
console.log('PASS');
