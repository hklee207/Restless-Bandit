// EV-charging restless bandit engine: a line-by-line port of envs/deadlineSchedulingEnv.py
// (authors' NeurWIN code) plus the top-M index policy from testDeadlineScheduling.py.
(function (root) {
  const C = 0.5, BETA = 0.99, LEN = 300, P_ARRIVE = 0.7;

  function rng(seed) { // mulberry32
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Per-spot random draws, same layout as the env: job[t] by time step, T[i]/B[i] by arrival count.
  function makeDraws(seed, N) {
    const out = [];
    for (let k = 0; k < N; k++) {
      const r = rng((seed * 1000003 + k * 7919 + 17) >>> 0);
      const d = { job: [], T: [], B: [] };
      for (let i = 0; i < LEN; i++) {
        d.job.push(r() < P_ARRIVE ? 1 : 0);
        d.T.push(1 + Math.floor(r() * 12));
        d.B.push(1 + Math.floor(r() * 9));
      }
      out.push(d);
    }
    return out;
  }

  function newStation(draws) {
    return {
      t: 0,
      spots: draws.map(d => ({ T: d.T[0], B: d.B[0], load: 1, draws: d })),
      units: 0, penalty: 0, net: 0, disc: 0, history: [0]
    };
  }

  function arrival(s, t) { // _newArrival(): jobList[episodeTime - 1], episodeTime = t + 1
    if (s.draws.job[t] === 1) { s.T = s.draws.T[s.load]; s.B = s.draws.B[s.load]; s.load += 1; return true; }
    s.T = 0; s.B = 0; return false;
  }

  // Top-M by score, ties to the lowest spot number (getSelection()).
  function choose(station, score, M) {
    const order = station.spots.map((s, k) => ({ k, v: score(s.T, s.B) }))
      .sort((a, b) => (b.v - a.v) || (a.k - b.k));
    return order.slice(0, M).map(o => o.k);
  }

  // One round. Returns picks, reward and events for narration.
  function step(station, score, M) {
    const t = station.t;
    const picks = choose(station, score, M);
    const pickSet = new Set(picks);
    let reward = 0;
    const events = [];
    station.spots.forEach((s, k) => {
      const act = pickSet.has(k);
      let r = 0;
      if (s.T === 0 && s.B === 0) {
        r = 0; if (arrival(s, t)) events.push({ k, type: 'arrive', T: s.T, B: s.B });
      } else if (s.T > 1) {
        if (act) {
          r = 1 - C; s.T -= 1; s.B -= 1;
          if (s.B < 0) { s.B = 0; r = 0; } else station.units += 1;
        } else { r = 0; s.T -= 1; }
      } else { // T == 1: car leaves after this round
        let short;
        if (act) {
          r = (1 - C) - 0.2 * Math.pow(s.B - 1, 2);
          if (s.B === 0) r = 0; else station.units += 1;
          short = Math.max(0, s.B - 1);
        } else { r = -0.2 * s.B * s.B; short = s.B; }
        const pen = 0.2 * short * short;
        station.penalty += pen;
        events.push({ k, type: 'leave', short, penalty: pen });
        s.T = 0; s.B = 0;
        if (arrival(s, t)) events.push({ k, type: 'arrive', T: s.T, B: s.B });
      }
      reward += r;
    });
    station.net += reward;
    station.disc += Math.pow(BETA, t) * reward;
    station.t += 1;
    station.history.push(station.net);
    return { picks, reward, events };
  }

  function closedForm(T, B) { // selectDeadlineIndex()
    if (B === 0) return 0;
    if (B >= 1 && B <= T - 1) return 1 - C;
    return Math.pow(BETA, T - 1) * 0.2 * Math.pow(B - T + 1, 2) - Math.pow(BETA, T - 1) * 0.2 * Math.pow(B - T, 2) + 1 - C;
  }

  function tableScore(tab) { return (T, B) => tab[T][B]; }

  function runEpisode(draws, score, M) {
    const st = newStation(draws);
    const rewards = [], picks = [];
    for (let i = 0; i < LEN; i++) { const o = step(st, score, M); rewards.push(o.reward); picks.push(o.picks); }
    return { disc: st.disc, rewards, picks };
  }

  // Paired comparison over many seeds (same arrivals for both policies).
  function compare(N, M, scoreA, scoreB, runs, seed0) {
    const a = [], b = [];
    for (let i = 0; i < runs; i++) {
      const d = makeDraws(seed0 + i, N);
      a.push(runEpisode(d, scoreA, M).disc);
      b.push(runEpisode(d, scoreB, M).disc);
    }
    const mean = x => x.reduce((s, v) => s + v, 0) / x.length;
    const diff = b.map((v, i) => v - a[i]), md = mean(diff);
    const sd = Math.sqrt(diff.reduce((s, v) => s + (v - md) ** 2, 0) / (diff.length - 1));
    return { meanA: mean(a), meanB: mean(b), diff: md, se: sd / Math.sqrt(runs), wins: diff.filter(v => v > 0).length, runs };
  }

  root.EV = { C, BETA, LEN, rng, makeDraws, newStation, step, choose, closedForm, tableScore, runEpisode, compare };
})(typeof window !== 'undefined' ? window : globalThis);
