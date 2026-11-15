/*
 * NeurWIN vs Whittle index simulator.
 *
 * Part 1 (SimCore) is a line-by-line port of envs/deadlineSchedulingEnv.py and of the
 * selection rule in testingScripts/testDeadlineScheduling.py. It has no page code, so
 * test_sim.mjs can load it in Node and check it against the Python env.
 * Part 2 is the page. It only runs in a browser.
 */
(function (root) {
  "use strict";

  // ───────────────────────────── Part 1: core ─────────────────────────────

  // Small seeded random generator (mulberry32). Only used to pre-draw arrivals.
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // One independent stream per (seed, spot), so spot k gets the same cars whatever N is.
  function spotSeed(seed, spot) {
    let h = (seed >>> 0) ^ Math.imul(spot + 1, 0x9E3779B1);
    h = Math.imul(h ^ (h >>> 16), 0x85EBCA6B);
    h = Math.imul(h ^ (h >>> 13), 0xC2B2AE35);
    return (h ^ (h >>> 16)) >>> 0;
  }

  /*
   * The random draws reset() makes for one spot, in the same order:
   *   jobList           = G.choice([1,0], p=[newJobProb, 1-newJobProb], size=episodeLimit)
   *   deadline          = G.randint(1, maxDeadline+1, size=episodeLimit)   (timeUntilDeadline is a copy)
   *   load              = G.randint(1, maxLoad+1, size=episodeLimit)
   * Same distributions as Python, different generator. (One spare entry is drawn so the
   * arrival counter can never run past the end of the list.)
   */
  function drawArrivals(seed, spot, episodeLimit, opt) {
    opt = opt || {};
    const p = opt.newJobProb === undefined ? 0.7 : opt.newJobProb;
    const maxDeadline = opt.maxDeadline || 12, maxLoad = opt.maxLoad || 9;
    const rng = mulberry32(spotSeed(seed, spot));
    const jobList = [], timeUntilDeadline = [], load = [];
    for (let i = 0; i < episodeLimit; i++) jobList.push(rng() < p ? 1 : 0);
    for (let i = 0; i <= episodeLimit; i++) timeUntilDeadline.push(1 + Math.floor(rng() * maxDeadline));
    for (let i = 0; i <= episodeLimit; i++) load.push(1 + Math.floor(rng() * maxLoad));
    return { jobList: jobList, timeUntilDeadline: timeUntilDeadline, load: load };
  }

  /*
   * One charging spot = one arm = one deadlineSchedulingEnv (noiseVar = 0, train = False).
   * State is (T, B): rounds until the car leaves, charge still needed. (0, 0) = empty.
   */
  function Spot(draws, processingCost) {
    this.jobList = draws.jobList;
    this.timeUntilDeadline = draws.timeUntilDeadline;
    this.load = draws.load;
    this.processingCost = processingCost;
    this.reset();
  }

  // reset(): the first car is always present; loadIndex then points at the next car.
  Spot.prototype.reset = function () {
    this.loadIndex = 0;
    this.T = this.timeUntilDeadline[0];
    this.B = this.load[0];
    this.episodeTime = 0;
    this.loadIndex += 1;
  };

  // _newArrival(): jobList is indexed by time step, deadline/load by arrival count.
  Spot.prototype._newArrival = function () {
    const job = this.jobList[this.episodeTime - 1];
    if (job === 1) {
      this.T = this.timeUntilDeadline[this.loadIndex];
      this.B = this.load[this.loadIndex];
      this.loadIndex += 1;
      return true;
    }
    this.T = 0;
    this.B = 0;
    return false;
  };

  // step(action) + _calRewardAndState(action). `reward` uses the env's own expressions;
  // the other fields only describe what happened, for the page.
  Spot.prototype.step = function (action) {
    this.episodeTime += 1;
    const T = this.T, B = this.B, c = this.processingCost;
    let reward = 0, earned = 0, penalty = 0, left = false, arrived = false;

    if (action === 1) {
      if (B === 0 && T === 0) {
        reward = 0;
        arrived = this._newArrival();
      } else if (B >= 0 && T > 1) {
        reward = (1 - c);
        this.T -= 1;
        this.B -= 1;
        if (this.B < 0) {
          this.B = 0;
          reward = 0;
        }
        earned = reward;
      } else if (B >= 0 && T === 1) {
        reward = ((1 - c) - 0.2 * ((B - 1) ** 2));
        if (B === 0) {
          reward = 0;
        } else {
          earned = 1 - c;
          penalty = 0.2 * ((B - 1) ** 2);
        }
        this.B = 0;
        this.T = 0;
        left = true;
        arrived = this._newArrival();
      }
    } else {
      if (B === 0 && T === 0) {
        reward = 0;
        arrived = this._newArrival();
      } else if (B >= 0 && T > 1) {
        reward = 0;
        this.T -= 1;
      } else if (B >= 0 && T === 1) {
        reward = -0.2 * (B ** 2);
        penalty = 0.2 * (B ** 2);
        this.B = 0;
        this.T = 0;
        left = true;
        arrived = this._newArrival();
      }
    }
    return { reward: reward, earned: earned, penalty: penalty, unit: earned > 0 ? 1 : 0,
             left: left, arrived: arrived, action: action, before: { T: T, B: B } };
  };

  // getSelection(): take the max M times; on a tie the first (lowest-numbered) arm wins.
  function getSelection(index, schedule) {
    const taken = new Array(index.length).fill(false), result = [];
    for (let i = 0; i < schedule; i++) {
      let best = -1;
      for (let k = 0; k < index.length; k++) {
        if (taken[k]) continue;
        if (best === -1 || index[k] > index[best]) best = k;
      }
      taken[best] = true;
      result.push(best);
    }
    return result;
  }

  // A station: N spots, power for M per round, and one index table table[T][B].
  function Station(table, drawsPerSpot, M, opt) {
    opt = opt || {};
    this.table = table;
    this.M = M;
    this.beta = opt.beta === undefined ? 0.99 : opt.beta;
    const c = opt.c === undefined ? 0.5 : opt.c;
    this.spots = drawsPerSpot.map(function (d) { return new Spot(d, c); });
    this.t = 0;
    this.units = 0; this.penalty = 0; this.net = 0; this.discounted = 0;
    this.last = null;
  }

  // The choice for the current state. Does not change anything.
  Station.prototype.decide = function () {
    const table = this.table;
    const index = this.spots.map(function (s) { return table[s.T][s.B]; });
    return { index: index, chosen: getSelection(index, this.M) };
  };

  // One round, in the test script's order: chosen arms first, then the rest by number.
  Station.prototype.step = function () {
    const d = this.decide(), outcomes = new Array(this.spots.length);
    const isChosen = new Array(this.spots.length).fill(false);
    let cumReward = 0;
    for (const arm of d.chosen) {
      isChosen[arm] = true;
      outcomes[arm] = this.spots[arm].step(1);
      cumReward += outcomes[arm].reward;
    }
    for (let k = 0; k < this.spots.length; k++) {
      if (isChosen[k]) continue;
      outcomes[k] = this.spots[k].step(0);
      cumReward += outcomes[k].reward;
    }
    for (const o of outcomes) { this.units += o.unit; this.penalty += o.penalty; }
    this.net += cumReward;
    this.discounted += (this.beta ** this.t) * cumReward;
    this.t += 1;
    this.last = { reward: cumReward, chosen: d.chosen, index: d.index, outcomes: outcomes };
    return this.last;
  };

  function tableFor(data, checkpoint) {
    const i = data.checkpoints.indexOf(checkpoint);
    if (i < 0) throw new Error("no checkpoint " + checkpoint);
    return data.neurwin[i];
  }

  // Run both policies on the same arrivals for each seed. Returns discounted totals.
  function runPaired(N, M, tableA, tableB, seeds, opt) {
    opt = opt || {};
    const L = opt.episodeLimit || 300, a = [], b = [];
    for (const seed of seeds) {
      const draws = [];
      for (let k = 0; k < N; k++) draws.push(drawArrivals(seed, k, L, opt));
      const sa = new Station(tableA, draws, M, opt), sb = new Station(tableB, draws, M, opt);
      for (let t = 0; t < L; t++) { sa.step(); sb.step(); }
      a.push(sa.discounted); b.push(sb.discounted);
    }
    return { a: a, b: b };
  }

  function mean(x) { let s = 0; for (const v of x) s += v; return s / x.length; }

  // Paired difference b - a, with its standard error (sample std / sqrt(n)).
  function pairedSummary(a, b) {
    const d = a.map(function (v, i) { return b[i] - v; }), m = mean(d);
    let ss = 0, better = 0;
    for (const v of d) { ss += (v - m) * (v - m); if (v > 0) better++; }
    const sd = d.length > 1 ? Math.sqrt(ss / (d.length - 1)) : 0;
    return { meanA: mean(a), meanB: mean(b), diff: m, se: sd / Math.sqrt(d.length), bBetter: better, n: d.length };
  }

  const SimCore = { mulberry32: mulberry32, drawArrivals: drawArrivals, Spot: Spot, Station: Station,
                    getSelection: getSelection, tableFor: tableFor, runPaired: runPaired,
                    pairedSummary: pairedSummary, mean: mean };
  if (typeof module !== "undefined" && module.exports) module.exports = SimCore;
  root.SimCore = SimCore;

  // ───────────────────────────── Part 2: page ─────────────────────────────
  if (typeof document === "undefined") return;

  const BLUE = "#2a78d6", ORANGE = "#eb6834", INK2 = "#6b6a64", GRID = "#e6e5e0";
  const MINUS = "−";
  function $(id) { return document.getElementById(id); }
  function num(x, digits) {
    const s = Math.abs(x).toFixed(digits === undefined ? 2 : digits);
    return (x < 0 && Number(s) !== 0 ? MINUS : "") + s;
  }
  function signed(x, digits) { return (x > 0 ? "+" : "") + num(x, digits); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (ch) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]; }); }

  document.addEventListener("DOMContentLoaded", function () {
    const D = root.SIM_DATA;
    if (!D) {
      $("fatal").hidden = false;
      $("fatal").textContent = "data.js is missing. Run export_data.py in this folder, then reload.";
      return;
    }
    const L = D.meta.episodeLimit, OPT = { beta: D.meta.beta, c: D.meta.c, episodeLimit: L, newJobProb: D.meta.newJobProb };
    const P = { N: 4, M: 1, seed: 1, speed: 4, ckpt: 2000 };
    let stIndex, stNeur, hist, differCount, timer = null;

    // ── simulation control ──
    function reset() {
      pause();
      const draws = [];
      for (let k = 0; k < P.N; k++) draws.push(drawArrivals(P.seed, k, L, OPT));
      stIndex = new Station(D.closedForm, draws, P.M, OPT);
      stNeur = new Station(tableFor(D, P.ckpt), draws, P.M, OPT);
      hist = { index: [0], neur: [0] };
      differCount = 0;
      render();
    }
    function sameChoice(a, b) {
      const x = a.slice().sort(), y = b.slice().sort();
      return x.length === y.length && x.every(function (v, i) { return v === y[i]; });
    }
    function stepOnce() {
      if (stIndex.t >= L) { pause(); return; }
      if (!sameChoice(stIndex.decide().chosen, stNeur.decide().chosen)) differCount++;
      stIndex.step(); stNeur.step();
      hist.index.push(stIndex.net); hist.neur.push(stNeur.net);
      if (stIndex.t >= L) pause();
    }
    function play() {
      if (stIndex.t >= L) return;
      pause();
      timer = setInterval(function () { stepOnce(); render(); }, 1000 / P.speed);
      $("play").textContent = "Pause";
    }
    function pause() {
      if (timer) clearInterval(timer);
      timer = null;
      $("play").textContent = "Play";
    }

    // ── stations ──
    const CAR = '<svg class="car" viewBox="0 0 48 22" aria-hidden="true"><path d="M2 16v-4a2 2 0 0 1 1.600-2l7.400-1.500 4-4.500a3 3 0 0 1 2.200-1h11.600a3 3 0 0 1 2.300 1.100l3.900 4.700 7.400 1.600a2 2 0 0 1 1.600 2V16a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1z"/><circle class="wheel" cx="13" cy="17" r="3.500"/><circle class="wheel" cx="35" cy="17" r="3.500"/></svg>';

    function outcomeText(o) {
      if (!o) return "";
      const b = o.before;
      if (b.T === 0 && b.B === 0) return o.arrived ? "new car arrived" : "";
      let s = "";
      if (o.left) {
        const missing = o.action === 1 ? Math.max(b.B - 1, 0) : b.B;
        s = missing > 0 ? "left " + missing + " short, paid " + num(o.penalty, 1) : "left fully charged";
        if (o.arrived) s += "; new car";
      } else if (o.unit) {
        s = "+" + num(o.earned, 1) + " charged";
      }
      return s;
    }

    function renderStation(el, st, dec, other) {
      const done = st.t >= L;
      let h = '<div class="spots">';
      st.spots.forEach(function (s, k) {
        const empty = s.T === 0 && s.B === 0;
        const charged = !done && dec.chosen.indexOf(k) >= 0;
        const risk = !empty && s.B >= s.T && s.B > 0;
        const diff = charged && other && other.indexOf(k) < 0;
        const o = st.last ? st.last.outcomes[k] : null;
        const note = outcomeText(o);
        h += '<div class="spot' + (empty ? " empty" : "") + (charged ? " charged" : "") + (risk ? " risk" : "") + (diff ? " differs" : "") + '">';
        h += '<div class="spot-head"><span>Spot ' + (k + 1) + "</span>" + (charged ? '<span class="tag">charging</span>' : "") + "</div>";
        if (empty) {
          h += '<div class="spot-body"><div class="empty-label">empty</div></div>';
        } else {
          h += '<div class="spot-body">' + CAR + '<div class="count"><b>' + s.T + "</b><span>round" + (s.T === 1 ? "" : "s") + " left</span></div></div>";
          h += '<div class="bbar" title="charge still needed">';
          for (let i = 1; i <= D.meta.maxB; i++) h += '<i class="' + (i <= s.B ? "on" : "") + '"></i>';
          h += '</div><div class="need">needs ' + s.B + (risk ? ' <em>at risk</em>' : "") + "</div>";
        }
        h += '<div class="idx">index ' + (done ? "–" : num(dec.index[k])) + "</div>";
        h += '<div class="note' + (o && o.penalty > 0 ? " bad" : "") + '">' + (note ? esc(note) : "&nbsp;") + "</div>";
        h += "</div>";
      });
      h += "</div>";
      el.innerHTML = h;
    }

    function renderStats(el, st) {
      const cells = [
        ["Round", st.t + " / " + L],
        ["Units charged", String(st.units)],
        ["Penalty paid", num(st.penalty, 1)],
        ["Net reward", num(st.net, 1)],
        ["Discounted total (β = " + D.meta.beta + ")", num(st.discounted, 1)],
      ];
      el.innerHTML = cells.map(function (c) { return "<div><span>" + c[0] + "</span><b>" + c[1] + "</b></div>"; }).join("");
    }

    function describe(s) { return s.T === 0 && s.B === 0 ? "empty" : "T=" + s.T + ", B=" + s.B; }
    function explain(st, dec) {
      if (st.t >= L) return "Episode finished after " + L + " rounds.";
      const ch = dec.chosen;
      if (ch.length === 1) {
        const k = ch[0], v = dec.index[k], ties = [];
        dec.index.forEach(function (x, j) { if (j !== k && x === v) ties.push(j + 1); });
        let s = "Charges spot " + (k + 1) + " (" + describe(st.spots[k]) + "): index " + num(v) + " is the highest";
        if (ties.length) s += ", tied with spot" + (ties.length > 1 ? "s " : " ") + ties.join(", ") + "; ties go to the lowest spot number";
        return s + ".";
      }
      const parts = ch.slice().sort(function (a, b) { return a - b; }).map(function (k) {
        return (k + 1) + " (" + describe(st.spots[k]) + ", " + num(dec.index[k]) + ")";
      });
      return "Charges spots " + parts.join(", ") + ": the " + ch.length + " highest indices.";
    }

    // ── charts ──
    function niceStep(span, target) {
      const raw = span / target, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
      return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
    }
    function path(xs, ys) {
      let s = "";
      for (let i = 0; i < xs.length; i++) s += (i ? "L" : "M") + xs[i].toFixed(1) + " " + ys[i].toFixed(1);
      return s;
    }

    function drawLive() {
      const W = 900, H = 230, ml = 56, mr = 96, mt = 12, mb = 34, pw = W - ml - mr, ph = H - mt - mb;
      let lo = 0, hi = 0;
      for (const v of hist.index) { if (v < lo) lo = v; if (v > hi) hi = v; }
      for (const v of hist.neur) { if (v < lo) lo = v; if (v > hi) hi = v; }
      if (hi - lo < 10) { lo -= 5; hi += 5; }
      const step = niceStep(hi - lo, 4);
      lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
      const X = function (t) { return ml + (t / L) * pw; }, Y = function (v) { return mt + ((hi - v) / (hi - lo)) * ph; };
      let s = "";
      for (let v = lo; v <= hi + 1e-9; v += step)
        s += '<line x1="' + ml + '" x2="' + (W - mr) + '" y1="' + Y(v) + '" y2="' + Y(v) + '" stroke="' + GRID + '"/>' +
             '<text x="' + (ml - 8) + '" y="' + (Y(v) + 4) + '" text-anchor="end" class="tick">' + num(v, 0) + "</text>";
      for (let t = 0; t <= L; t += 50)
        s += '<text x="' + X(t) + '" y="' + (H - mb + 18) + '" text-anchor="middle" class="tick">' + t + "</text>";
      s += '<text x="' + (ml + pw / 2) + '" y="' + (H - 2) + '" text-anchor="middle" class="axis">Round</text>';
      const ts = hist.index.map(function (_, i) { return X(i); });
      s += '<path d="' + path(ts, hist.index.map(Y)) + '" fill="none" stroke="' + ORANGE + '" stroke-width="2.5"/>';
      s += '<path d="' + path(ts, hist.neur.map(Y)) + '" fill="none" stroke="' + BLUE + '" stroke-width="2.5"/>';
      const n = hist.index.length - 1, yi = Y(hist.index[n]), yn = Y(hist.neur[n]);
      let li = yi + 4, ln = yn + 4;
      if (Math.abs(li - ln) < 15) { if (hist.neur[n] >= hist.index[n]) { ln = Math.min(li, ln) - 4; li = ln + 15; } else { li = Math.min(li, ln) - 4; ln = li + 15; } }
      s += '<text x="' + (X(n) + 8) + '" y="' + li + '" fill="' + ORANGE + '" class="endlab">Index ' + num(hist.index[n], 1) + "</text>";
      s += '<text x="' + (X(n) + 8) + '" y="' + ln + '" fill="' + BLUE + '" class="endlab">NeurWIN ' + num(hist.neur[n], 1) + "</text>";
      $("live").innerHTML = s;
    }

    const TW = 900, TH = 300, tl = 56, tr = 18, tt = 14, tb = 40, tpw = TW - tl - tr, tph = TH - tt - tb;
    const YMIN = -350, YMAX = -50, EPMAX = D.checkpoints[D.checkpoints.length - 1];
    const EPSTEP = D.checkpoints[1] - D.checkpoints[0];
    function TX(ep) { return tl + (ep / EPMAX) * tpw; }
    function TY(v) { return tt + ((YMAX - Math.max(YMIN, Math.min(YMAX, v))) / (YMAX - YMIN)) * tph; }

    function drawTraining() {
      const c = D.curve, xs = c.episode.map(TX);
      let s = "";
      for (let v = YMIN; v <= YMAX; v += 50)
        s += '<line x1="' + tl + '" x2="' + (TW - tr) + '" y1="' + TY(v) + '" y2="' + TY(v) + '" stroke="' + GRID + '"/>' +
             '<text x="' + (tl - 8) + '" y="' + (TY(v) + 4) + '" text-anchor="end" class="tick">' + num(v, 0) + "</text>";
      for (let ep = 0; ep <= EPMAX; ep += 250)
        s += '<text x="' + TX(ep) + '" y="' + (TH - tb + 18) + '" text-anchor="middle" class="tick">' + ep + "</text>";
      s += '<text x="' + (tl + tpw / 2) + '" y="' + (TH - 4) + '" text-anchor="middle" class="axis">Training episodes</text>';
      s += '<text transform="translate(14 ' + (tt + tph / 2) + ') rotate(-90)" text-anchor="middle" class="axis">Total discounted reward</text>';
      const up = path(xs, c.p95.map(TY)), rev = c.p05.map(TY).reverse(), rx = xs.slice().reverse();
      let band = up;
      for (let i = 0; i < rx.length; i++) band += "L" + rx[i].toFixed(1) + " " + rev[i].toFixed(1);
      s += '<path d="' + band + 'Z" fill="' + BLUE + '" opacity="0.15"/>';
      s += '<path d="' + path(xs, c.mean.map(TY)) + '" fill="none" stroke="' + BLUE + '" stroke-width="2.5"/>';
      s += '<line x1="' + tl + '" x2="' + (TW - tr) + '" y1="' + TY(D.indexMean) + '" y2="' + TY(D.indexMean) + '" stroke="' + ORANGE + '" stroke-width="2.5" stroke-dasharray="8 6"/>';
      s += '<text x="' + (TW - tr - 4) + '" y="' + (TY(D.indexMean) + 18) + '" text-anchor="end" fill="' + ORANGE + '" class="endlab">Closed-form Whittle index ' + num(D.indexMean, 1) + "</text>";
      s += '<text x="' + (TW - tr - 4) + '" y="' + (TY(c.mean[c.mean.length - 1]) - 10) + '" text-anchor="end" fill="' + BLUE + '" class="endlab">NeurWIN, mean of 50 test runs (band: 5th to 95th percentile)</text>';
      s += '<g id="marker"></g>';
      $("training").innerHTML = s;
      drawMarker();
    }
    function drawMarker() {
      const i = D.checkpoints.indexOf(P.ckpt), x = TX(P.ckpt), y = TY(D.curve.mean[i]);
      $("marker").innerHTML =
        '<line x1="' + x + '" x2="' + x + '" y1="' + tt + '" y2="' + (tt + tph) + '" stroke="#1f1f1e" stroke-width="1.5"/>' +
        '<circle cx="' + x + '" cy="' + y + '" r="7" fill="#fff" stroke="' + BLUE + '" stroke-width="3"/>' +
        '<rect x="' + (x - 14) + '" y="' + tt + '" width="28" height="' + tph + '" fill="transparent" class="grab"/>';
      $("ckptLabel").innerHTML = "Checkpoint driving the NeurWIN station: <b>" + P.ckpt + " training episodes</b>" +
        (P.ckpt === 0 ? " (untrained network)" : "") + ". Mean test reward " + num(D.curve.mean[i], 1) + ".";
      $("ckptRange").value = P.ckpt;
      $("neurTitle").textContent = "NeurWIN (trained net, " + P.ckpt + " episodes)";
    }
    function setCheckpoint(ep) {
      ep = Math.max(0, Math.min(EPMAX, Math.round(ep / EPSTEP) * EPSTEP));
      if (ep === P.ckpt) return;
      P.ckpt = ep;
      $("batchOut").textContent = "";
      drawMarker();
      reset();
    }

    // ── render ──
    function render() {
      const dI = stIndex.decide(), dN = stNeur.decide(), done = stIndex.t >= L;
      const differ = !done && !sameChoice(dI.chosen, dN.chosen);
      renderStation($("spotsIndex"), stIndex, dI, differ ? dN.chosen : null);
      renderStation($("spotsNeur"), stNeur, dN, differ ? dI.chosen : null);
      renderStats($("statsIndex"), stIndex);
      renderStats($("statsNeur"), stNeur);
      $("whyIndex").textContent = explain(stIndex, dI);
      $("whyNeur").textContent = explain(stNeur, dN);
      const banner = $("differ");
      banner.className = "differ" + (differ ? " on" : "");
      banner.innerHTML = (differ ? "<b>Different choice this round.</b> " : "Same choice this round. ") +
        "The two stations chose differently in <b>" + differCount + "</b> of " + stIndex.t + " rounds played.";
      drawLive();
    }

    // ── controls ──
    function syncM() {
      $("M").max = P.N;
      if (P.M > P.N) P.M = P.N;
      $("M").value = P.M;
      $("Nval").textContent = P.N;
      $("Mval").textContent = P.M;
    }
    $("N").addEventListener("input", function () { P.N = +this.value; syncM(); $("batchOut").textContent = ""; reset(); });
    $("M").addEventListener("input", function () { P.M = +this.value; syncM(); $("batchOut").textContent = ""; reset(); });
    $("seed").addEventListener("change", function () {
      P.seed = Math.max(0, Math.floor(+this.value || 0)); this.value = P.seed; $("batchOut").textContent = ""; reset();
    });
    $("speed").addEventListener("input", function () { P.speed = +this.value; $("speedVal").textContent = P.speed; if (timer) play(); });
    $("play").addEventListener("click", function () { if (timer) pause(); else play(); });
    $("step").addEventListener("click", function () { pause(); stepOnce(); render(); });
    $("reset").addEventListener("click", reset);
    $("ckptRange").addEventListener("input", function () { setCheckpoint(+this.value); });

    const svg = $("training");
    let dragging = false;
    function epAt(ev) {
      const r = svg.getBoundingClientRect(), px = ((ev.clientX - r.left) / r.width) * TW;
      return ((px - tl) / tpw) * EPMAX;
    }
    svg.addEventListener("pointerdown", function (ev) { dragging = true; svg.setPointerCapture(ev.pointerId); setCheckpoint(epAt(ev)); ev.preventDefault(); });
    svg.addEventListener("pointermove", function (ev) { if (dragging) setCheckpoint(epAt(ev)); });
    svg.addEventListener("pointerup", function () { dragging = false; });
    svg.addEventListener("pointercancel", function () { dragging = false; });

    $("batch").addEventListener("click", function () {
      const seeds = [];
      for (let i = 0; i < 50; i++) seeds.push(P.seed + i);
      const r = runPaired(P.N, P.M, D.closedForm, tableFor(D, P.ckpt), seeds, OPT), s = pairedSummary(r.a, r.b);
      $("batchOut").innerHTML =
        "N = " + P.N + ", M = " + P.M + ", checkpoint " + P.ckpt + ", seeds " + seeds[0] + " to " + seeds[49] + " (discounted totals):<br>" +
        '<span class="c-index">Index mean <b>' + num(s.meanA, 1) + "</b></span> &nbsp; " +
        '<span class="c-neur">NeurWIN mean <b>' + num(s.meanB, 1) + "</b></span> &nbsp; " +
        "Paired difference (NeurWIN " + MINUS + " index) <b>" + signed(s.diff, 2) + "</b> ± " + num(s.se, 2) +
        " (standard error). NeurWIN better in " + s.bBetter + " of " + s.n + ".";
    });

    // ── scale table ──
    $("scaleBody").innerHTML = D.scale.map(function (r) {
      return "<tr><td>" + r.N + "</td><td>" + r.M + "</td><td>" + num(r.neurwin, 1) + "</td><td>" + num(r.index, 1) +
        "</td><td>" + signed(r.gap, 1) + "</td><td>" + signed(r.gap_pct, 1) + "%</td><td>" + r.neurwin_better_runs + " / 50</td></tr>";
    }).join("");

    $("ckptRange").max = EPMAX; $("ckptRange").step = EPSTEP;
    $("episodeLen").textContent = L;
    syncM();
    drawTraining();
    reset();

    // Testing aid: index.html?steps=40 fast-forwards 40 rounds on load.
    const pre = +new URLSearchParams(location.search).get("steps") || 0;
    if (pre > 0) { for (let i = 0; i < pre; i++) stepOnce(); render(); }
  });
})(typeof globalThis !== "undefined" ? globalThis : this);
