/*
 * app.js — UI, training loop and rendering for the Neural Net Playground.
 * Depends on nn.js, datasets.js, presets.js, plots.js (and KaTeX if available).
 */
(function () {
  'use strict';
  var NN = window.NN, DS = window.Datasets, PRESETS = window.Presets.PRESETS, BASE = window.Presets.BASE, Plots = window.Plots;

  /* ---------- Constants ---------- */
  var G = 60;                       // heatmap grid: G x G cells, (G+1)^2 corner samples
  var NG = G + 1;                   // samples per side (cell corners, incl. domain edges)
  var D = DS.DOMAIN;                // plot domain [-D, D]^2
  var N_POINTS = 400;
  var LR_MIN = 1e-4, LR_MAX = 3;
  var L2_VALUES = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1];
  var BATCH_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 0];   // 0 = full batch
  var SPEED_VALUES = [1 / 8, 1 / 4, 1 / 2, 1, 2, 3, 5, 10, 25];
  var MAX_LAYERS = 4, MAX_NEURONS = 8;
  var FRAME_BUDGET_MS = 22;

  var COL = {
    c0: [217, 119, 46], c1: [47, 109, 181],
    h0: [238, 178, 132], hm: [247, 246, 243], h1: [140, 176, 222],   // heatmap (light)
    t0: [224, 136, 70], t1: [66, 122, 192]                            // thumbnails (stronger)
  };
  var LAYER_COLORS = ['#1f5f99', '#5b8fc0', '#93b5d6', '#9a9a9f', '#c7c7c2'];

  /* ---------- State ---------- */
  var S = {
    dataset: BASE.dataset, noise: BASE.noise, trainFrac: BASE.trainFrac,
    hidden: BASE.hidden.slice(), activation: BASE.activation, optimizer: BASE.optimizer,
    lr: BASE.lr, l2: BASE.l2, batch: BASE.batch, speed: BASE.speed,
    dataSeed: BASE.dataSeed, initSeed: BASE.initSeed, compareSGD: BASE.compareSGD,
    data: null, split: null, net: null, opt: null, cmpNet: null, cmpOpt: null,
    order: null, orderRng: null, epoch: 0, hist: null, last: null,
    playing: false, acc: 0, diverged: false, dirty: true, dead: null,
    maps: null, inputMaps: null, hover: null, tileRects: [], presetId: null,
    showTest: true, discretize: false, logLoss: true,
    watchDismissed: false, watchMode: null   // 'preset' | 'tried' | 'neutral'
  };

  var $ = function (id) { return document.getElementById(id); };
  var el = {};
  ['presetList', 'presetDesc', 'dataset', 'noise', 'noiseOut', 'split', 'splitOut', 'regen', 'layersOut', 'layersDec', 'layersInc',
    'layerSteppers', 'activation', 'optimizer', 'compareSGD', 'lr', 'lrOut', 'l2', 'l2Out', 'batch', 'batchOut', 'play', 'step', 'reset',
    'speed', 'speedOut', 'rEpoch', 'rTrainLoss', 'rTestLoss', 'rTrainAcc', 'rTestAcc', 'boundary', 'status', 'showTest', 'discretize',
    'meta', 'network', 'netInfo', 'lossPlot', 'gradPlot', 'logLoss', 'lossLegend', 'gradLegend', 'eqAct', 'eqOpt', 'eqInit', 'optName',
    'watch', 'watchTag', 'watchPreset', 'watchClose', 'watchText', 'watchTry'
  ].forEach(function (id) { el[id] = $(id); });

  /* ---------- Helpers ---------- */
  var lrToSlider = function (lr) { return Math.round(1000 * Math.log(lr / LR_MIN) / Math.log(LR_MAX / LR_MIN)); };
  var sliderToLr = function (s) { return +(LR_MIN * Math.pow(LR_MAX / LR_MIN, s / 1000)).toPrecision(2); };
  function nearestIndex(arr, v) {
    var best = 0, bd = Infinity;
    for (var i = 0; i < arr.length; i++) { var d = Math.abs(arr[i] - v); if (d < bd) { bd = d; best = i; } }
    return best;
  }
  function fmtNum(v) {
    if (v == null || !isFinite(v)) return '–';
    if (v === 0) return '0';
    if (Math.abs(v) < 1e-3) return v.toExponential(2);
    return v.toFixed(4);
  }
  function fmtLr(v) { return v >= 0.01 ? String(+v.toPrecision(2)) : v.toExponential(1).replace('e-', 'e−'); }
  function fmtPct(v) { return v == null || !isFinite(v) ? '–' : (100 * v).toFixed(1) + '%'; }
  function batchSize() { var n = S.split.train.length; return S.batch > 0 ? Math.min(S.batch, n) : n; }
  function lerp3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

  // value in [0,1] -> diverging color through a neutral midpoint
  function diverge(t, lo, mid, hi) {
    if (t < 0) t = 0; else if (t > 1) t = 1;
    return t < 0.5 ? lerp3(lo, mid, t * 2) : lerp3(mid, hi, (t - 0.5) * 2);
  }

  /* ---------- Data / network lifecycle ---------- */
  function buildData() {
    S.data = DS.generate(S.dataset, N_POINTS, S.noise, S.dataSeed);
    S.split = DS.split(N_POINTS, S.trainFrac, S.dataSeed);
  }

  function resetTraining() {
    S.net = new NN.MLP({ hidden: S.hidden, activation: S.activation, seed: S.initSeed });
    S.opt = new NN.Optimizer(S.net, S.optimizer);
    if (S.compareSGD) {
      S.cmpNet = new NN.MLP({ hidden: S.hidden, activation: S.activation, seed: S.initSeed });
      S.cmpNet.copyWeightsFrom(S.net);
      S.cmpOpt = new NN.Optimizer(S.cmpNet, 'sgd');
    } else { S.cmpNet = null; S.cmpOpt = null; }
    S.order = S.split.train.slice();
    S.orderRng = NN.mulberry32(S.initSeed * 7919 + 1);
    S.epoch = 0; S.acc = 0; S.diverged = false;
    S.hist = { trL: [], teL: [], cmpL: [], grad: [] };
    for (var l = 0; l < S.net.L; l++) S.hist.grad.push([]);
    var tr = S.net.evaluate(S.data.X, S.data.Y, S.split.train), te = S.net.evaluate(S.data.X, S.data.Y, S.split.test);
    S.last = { trL: tr.loss, teL: te.loss, trA: tr.acc, teA: te.acc };
    el.status.hidden = true;
    S.dirty = true;
    renderLegends();
  }

  function trainOneEpoch() {
    if (S.diverged) return false;
    var X = S.data.X, Y = S.data.Y;
    NN.shuffleInPlace(S.order, S.orderRng);
    var bs = batchSize();
    var r = NN.trainEpoch(S.net, S.opt, X, Y, S.order, bs, S.lr, S.l2);
    if (S.cmpNet) NN.trainEpoch(S.cmpNet, S.cmpOpt, X, Y, S.order, bs, S.lr, S.l2);
    S.epoch++;
    var tr = S.net.evaluate(X, Y, S.split.train), te = S.net.evaluate(X, Y, S.split.test);
    S.last = { trL: tr.loss, teL: te.loss, trA: tr.acc, teA: te.acc };
    S.hist.trL.push(tr.loss); S.hist.teL.push(te.loss);
    if (S.cmpNet) S.hist.cmpL.push(S.cmpNet.evaluate(X, Y, S.split.train).loss);
    for (var l = 0; l < r.gradNorms.length; l++) S.hist.grad[l].push(r.gradNorms[l]);
    if (!S.net.isFinite() || !isFinite(tr.loss)) {
      S.diverged = true; setPlaying(false);
      el.status.textContent = 'Diverged: parameters became non-finite. Lower η and reset.';
      el.status.hidden = false;
      return false;
    }
    S.dirty = true;
    return true;
  }

  /* ---------- Grid evaluation ---------- */
  function computeMaps() {
    var net = S.net, L = net.L, sizes = net.sizes;
    if (!S.maps || S.maps.key !== sizes.join('-')) {
      S.maps = { key: sizes.join('-'), out: new Float32Array(NG * NG), logit: new Float32Array(NG * NG), hidden: [] };
      for (var l = 0; l < L - 1; l++) {
        var row = [];
        for (var j = 0; j < sizes[l + 1]; j++) row.push(new Float32Array(NG * NG));
        S.maps.hidden.push(row);
      }
    }
    var M = S.maps, step = 2 * D / G;
    // samples sit on cell corners: (gx, gy) -> (-D + gx*step, D - gy*step), gx, gy = 0..G
    for (var gy = 0; gy < NG; gy++) {
      var y = D - gy * step;                    // row 0 at the top
      for (var gx = 0; gx < NG; gx++) {
        var x = -D + gx * step, p = gy * NG + gx;
        M.logit[p] = net.forward(x, y);
        M.out[p] = net.as[L][0];
        for (var l2 = 0; l2 < L - 1; l2++) {
          var a = net.as[l2 + 1], hr = M.hidden[l2];
          for (var j2 = 0; j2 < a.length; j2++) hr[j2][p] = a[j2];
        }
      }
    }
    if (!S.inputMaps) {
      S.inputMaps = [new Float32Array(NG * NG), new Float32Array(NG * NG)];
      for (var gy2 = 0; gy2 < NG; gy2++) for (var gx2 = 0; gx2 < NG; gx2++) {
        S.inputMaps[0][gy2 * NG + gx2] = (-D + gx2 * step) / D;
        S.inputMaps[1][gy2 * NG + gx2] = (D - gy2 * step) / D;
      }
    }
    S.dead = S.activation === 'relu' ? NN.deadUnits(net, S.data.X, S.split.train) : null;
  }

  var off = document.createElement('canvas');
  off.width = NG; off.height = NG;
  var offCtx = off.getContext('2d');
  var offImg = offCtx.createImageData(NG, NG);

  // Main heatmap: rendered at (roughly) display resolution by bilinearly
  // interpolating the network's LOGIT on the corner grid and only then
  // applying the sigmoid + colormap. Interpolating in logit space keeps the
  // p = 0.5 transition sharp and smooth even when the net is very confident
  // (interpolating saturated probabilities produces blocky staircases).
  var hm = document.createElement('canvas');
  var hmCtx = hm.getContext('2d');
  var hmImg = null;
  var LUT_N = 1024, LUT = new Uint8ClampedArray(LUT_N * 3), LUT_D = new Uint8ClampedArray(LUT_N * 3);
  (function buildLut() {
    for (var i = 0; i < LUT_N; i++) {
      var t = i / (LUT_N - 1), c = diverge(t, COL.h0, COL.hm, COL.h1), cd = diverge(t > 0.5 ? 0.8 : 0.2, COL.h0, COL.hm, COL.h1);
      LUT[3 * i] = c[0]; LUT[3 * i + 1] = c[1]; LUT[3 * i + 2] = c[2];
      LUT_D[3 * i] = cd[0]; LUT_D[3 * i + 1] = cd[1]; LUT_D[3 * i + 2] = cd[2];
    }
  })();
  function paintHeatmap(res, discretize) {
    if (hm.width !== res) { hm.width = res; hm.height = res; hmImg = hmCtx.createImageData(res, res); }
    var d = hmImg.data, Z = S.maps.logit, lut = discretize ? LUT_D : LUT, k = G / res;
    for (var py = 0; py < res; py++) {
      var v = (py + 0.5) * k, iy = Math.min(G - 1, v | 0), fy = v - iy, r0 = iy * NG, r1 = r0 + NG;
      for (var px = 0; px < res; px++) {
        var u = (px + 0.5) * k, ix = Math.min(G - 1, u | 0), fx = u - ix;
        var z0 = Z[r0 + ix] + (Z[r0 + ix + 1] - Z[r0 + ix]) * fx;
        var z1 = Z[r1 + ix] + (Z[r1 + ix + 1] - Z[r1 + ix]) * fx;
        var z = z0 + (z1 - z0) * fy;
        var p = z >= 0 ? 1 / (1 + Math.exp(-z)) : 1 - 1 / (1 + Math.exp(z));
        var li = 3 * ((p * (LUT_N - 1) + 0.5) | 0), o = 4 * (py * res + px);
        d[o] = lut[li]; d[o + 1] = lut[li + 1]; d[o + 2] = lut[li + 2]; d[o + 3] = 255;
      }
    }
    hmCtx.putImageData(hmImg, 0, 0);
    return hm;
  }

  // Marching squares on the logit grid: segments of the z = 0 (p = 0.5) contour,
  // with linear interpolation along cell edges. Returns [x0,y0,x1,y1,...] in grid units.
  function boundarySegments() {
    var Z = S.maps.logit, segs = [];
    function lerpPt(ax, ay, za, bx, by, zb) { var t = za / (za - zb); return [ax + (bx - ax) * t, ay + (by - ay) * t]; }
    for (var gy = 0; gy < G; gy++) for (var gx = 0; gx < G; gx++) {
      var a = Z[gy * NG + gx], b = Z[gy * NG + gx + 1], c = Z[(gy + 1) * NG + gx + 1], dd = Z[(gy + 1) * NG + gx];
      var pts = [];
      if ((a > 0) !== (b > 0)) pts.push(lerpPt(gx, gy, a, gx + 1, gy, b));
      if ((b > 0) !== (c > 0)) pts.push(lerpPt(gx + 1, gy, b, gx + 1, gy + 1, c));
      if ((c > 0) !== (dd > 0)) pts.push(lerpPt(gx + 1, gy + 1, c, gx, gy + 1, dd));
      if ((dd > 0) !== (a > 0)) pts.push(lerpPt(gx, gy + 1, dd, gx, gy, a));
      if (pts.length === 2) segs.push(pts[0][0], pts[0][1], pts[1][0], pts[1][1]);
      else if (pts.length === 4) {
        // saddle: disambiguate with the cell-centre value
        var centre = (a + b + c + dd) / 4;
        if ((centre > 0) === (a > 0)) { segs.push(pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1], pts[3][0], pts[3][1]); }
        else { segs.push(pts[0][0], pts[0][1], pts[3][0], pts[3][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1]); }
      }
    }
    return segs;
  }

  // Paint a G x G map into the offscreen canvas.
  // kind: 'prob' (values in [0,1]) or 'signed' (values roughly in [-1,1]).
  function paintMap(map, kind, opts) {
    opts = opts || {};
    var d = offImg.data, lo = opts.lo || COL.h0, hi = opts.hi || COL.h1, mid = COL.hm;
    for (var p = 0; p < NG * NG; p++) {
      var t = kind === 'prob' ? map[p] : 0.5 + 0.5 * map[p] * (opts.scale || 1);
      if (opts.discretize) t = t > 0.5 ? 0.8 : 0.2;
      var c = diverge(t, lo, mid, hi);
      d[4 * p] = c[0]; d[4 * p + 1] = c[1]; d[4 * p + 2] = c[2]; d[4 * p + 3] = 255;
    }
    offCtx.putImageData(offImg, 0, 0);
    return off;
  }

  function unitScale(map) {
    // tanh: already in [-1,1]; sigmoid: 2a-1 handled by offset; ReLU-family: normalize by max |a|
    var act = S.activation;
    if (act === 'tanh') return { kind: 'signed', scale: 1, shift: 0 };
    if (act === 'sigmoid') return { kind: 'prob', scale: 1 };
    var m = 1e-9;
    for (var p = 0; p < map.length; p++) { var v = Math.abs(map[p]); if (v > m) m = v; }
    return { kind: 'signed', scale: 1 / m };
  }

  /* ---------- Rendering: decision boundary ---------- */
  function drawBoundary() {
    var f = Plots.fitCanvas(el.boundary), ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);
    var hv = S.hover, src, label = null;
    if (hv && hv.col === 0) {
      src = paintMap(S.inputMaps[hv.j], 'signed', { lo: COL.t0, hi: COL.t1 });
      label = 'Input x' + (hv.j === 0 ? '₁' : '₂');
    } else if (hv && hv.col > 0 && hv.col < S.net.L && S.maps.hidden[hv.col - 1] && S.maps.hidden[hv.col - 1][hv.j]) {
      var m = S.maps.hidden[hv.col - 1][hv.j], us = unitScale(m);
      src = paintMap(m, us.kind, { scale: us.scale, lo: COL.t0, hi: COL.t1 });
      label = 'Hidden layer ' + hv.col + ', unit ' + (hv.j + 1) + ' activation';
    } else {
      // ~1 heatmap pixel per CSS pixel (capped for speed); smoothing upscales the rest
      src = paintHeatmap(Math.max(120, Math.min(480, Math.round(W))), S.discretize);
    }
    ctx.save();
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, 0, 0, W, H);
    ctx.restore();
    if (!label) {
      // crisp vector decision boundary (p = 0.5)
      var segs = boundarySegments(), gxs = W / G, gys = H / G;
      ctx.save();
      ctx.strokeStyle = 'rgba(40,40,46,0.55)'; ctx.lineWidth = 1.4; ctx.lineCap = 'round';
      ctx.beginPath();
      for (var sI = 0; sI < segs.length; sI += 4) {
        ctx.moveTo(segs[sI] * gxs, segs[sI + 1] * gys); ctx.lineTo(segs[sI + 2] * gxs, segs[sI + 3] * gys);
      }
      ctx.stroke();
      ctx.restore();
    }

    var sx = function (x) { return (x + D) / (2 * D) * W; };
    var sy = function (y) { return (D - y) / (2 * D) * H; };
    // light axes
    ctx.strokeStyle = 'rgba(0,0,0,0.08)'; ctx.lineWidth = 1;
    [-1, 0, 1].forEach(function (v) {
      ctx.beginPath(); ctx.moveTo(Math.round(sx(v)) + 0.5, 0); ctx.lineTo(Math.round(sx(v)) + 0.5, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, Math.round(sy(v)) + 0.5); ctx.lineTo(W, Math.round(sy(v)) + 0.5); ctx.stroke();
    });
    ctx.fillStyle = 'rgba(0,0,0,0.42)';
    ctx.font = '10.5px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    [-1, 0, 1].forEach(function (v) { ctx.fillText(String(v), sx(v) + 3, H - 15); });
    ctx.textBaseline = 'bottom';
    [-1, 1].forEach(function (v) { ctx.fillText(String(v), 4, sy(v) - 2); });

    // points
    var X = S.data.X, Y = S.data.Y, r = Math.max(2.6, Math.min(5, W / 150));
    var c0 = 'rgb(' + COL.c0.join(',') + ')', c1 = 'rgb(' + COL.c1.join(',') + ')';
    var tr = S.split.train, te = S.split.test, k, i;
    ctx.lineWidth = 1;
    for (i = 0; i < tr.length; i++) {
      k = tr[i];
      ctx.beginPath(); ctx.arc(sx(X[2 * k]), sy(X[2 * k + 1]), r, 0, 2 * Math.PI);
      ctx.fillStyle = Y[k] ? c1 : c0; ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.stroke();
    }
    if (S.showTest) {
      ctx.lineWidth = 1.7;
      for (i = 0; i < te.length; i++) {
        k = te[i];
        ctx.beginPath(); ctx.arc(sx(X[2 * k]), sy(X[2 * k + 1]), r, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(255,255,255,0.75)'; ctx.fill();
        ctx.strokeStyle = Y[k] ? c1 : c0; ctx.stroke();
      }
    }
    if (label) {
      ctx.font = '12px ' + getComputedStyle(document.body).fontFamily;
      var tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(255,255,255,0.92)'; ctx.fillRect(8, 8, tw + 14, 22);
      ctx.strokeStyle = '#cfcfca'; ctx.lineWidth = 1; ctx.strokeRect(8.5, 8.5, tw + 13, 21);
      ctx.fillStyle = '#1c1c1e'; ctx.textBaseline = 'middle'; ctx.fillText(label, 15, 19);
    }
  }

  /* ---------- Rendering: network diagram ---------- */
  function drawNetwork() {
    var f = Plots.fitCanvas(el.network), ctx = f.ctx, W = f.w, H = f.h;
    ctx.clearRect(0, 0, W, H);
    var net = S.net, sizes = net.sizes, cols = sizes.length;
    var maxN = Math.max.apply(null, sizes);
    var top = 22, bottom = 18;
    var T = Math.max(14, Math.min(38, (H - top - bottom) / maxN - 6, (W - 20) / (cols * 1.9)));
    var gap = Math.min(10, Math.max(4, ((H - top - bottom) - maxN * T) / Math.max(1, maxN - 1)));
    var xs = [];
    var marginL = T / 2 + 20, marginR = T / 2 + 8;   // left margin leaves room for x1/x2 labels
    for (var c = 0; c < cols; c++) xs.push(marginL + (cols === 1 ? 0 : c * (W - marginL - marginR) / (cols - 1)));
    var ys = sizes.map(function (n) {
      var tot = n * T + (n - 1) * gap, y0 = top + ((H - top - bottom) - tot) / 2 + T / 2, a = [];
      for (var j = 0; j < n; j++) a.push(y0 + j * (T + gap));
      return a;
    });
    var dead = S.dead;
    var isDead = function (col, j) { return !!(dead && col > 0 && col < cols - 1 && dead[col - 1][j]); };

    // edges
    // Edge width/alpha are normalized PER LAYER by the 90th percentile of |w|
    // (capped at 1), so the diagram stays readable whether weights are 0.1 or 30
    // and one large weight cannot shrink every other edge to nothing.
    var refW = net.W.map(function (Wl) {
      var a = Array.prototype.map.call(Wl, Math.abs).sort(function (x, y) { return x - y; });
      return Math.max(1e-9, a[Math.min(a.length - 1, Math.floor(0.9 * (a.length - 1)))]);
    });
    ctx.lineCap = 'round';
    for (var l2 = 0; l2 < net.L; l2++) {
      var nIn = sizes[l2], nOut = sizes[l2 + 1], Wl = net.W[l2];
      var x0 = xs[l2] + T / 2, x1 = xs[l2 + 1] - T / 2, cx = (x0 + x1) / 2;
      for (var j = 0; j < nOut; j++) for (var i = 0; i < nIn; i++) {
        var w = Wl[j * nIn + i], m = Math.min(1, Math.abs(w) / refW[l2]);
        var dim = isDead(l2, i) || isDead(l2 + 1, j);
        ctx.strokeStyle = w >= 0 ? 'rgb(' + COL.c1.join(',') + ')' : 'rgb(' + COL.c0.join(',') + ')';
        ctx.globalAlpha = (dim ? 0.08 : 0.15 + 0.55 * m);
        ctx.lineWidth = 0.4 + 2.6 * m;
        ctx.beginPath();
        ctx.moveTo(x0, ys[l2][i]);
        ctx.bezierCurveTo(cx, ys[l2][i], cx, ys[l2 + 1][j], x1, ys[l2 + 1][j]);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // nodes (thumbnails)
    S.tileRects = [];
    ctx.imageSmoothingEnabled = true;
    for (var c2 = 0; c2 < cols; c2++) {
      for (var j2 = 0; j2 < sizes[c2]; j2++) {
        var x = xs[c2] - T / 2, y = ys[c2][j2] - T / 2, src;
        if (c2 === 0) src = paintMap(S.inputMaps[j2], 'signed', { lo: COL.t0, hi: COL.t1 });
        else if (c2 === cols - 1) src = paintMap(S.maps.out, 'prob', { lo: COL.t0, hi: COL.t1 });
        else {
          var mp = S.maps.hidden[c2 - 1][j2], us = unitScale(mp);
          src = paintMap(mp, us.kind, { scale: us.scale, lo: COL.t0, hi: COL.t1 });
        }
        var dd = isDead(c2, j2);
        ctx.globalAlpha = dd ? 0.22 : 1;
        ctx.drawImage(src, x, y, T, T);
        ctx.globalAlpha = 1;
        var hov = S.hover && S.hover.col === c2 && S.hover.j === j2;
        ctx.lineWidth = hov ? 2 : 1;
        ctx.strokeStyle = hov ? '#1f5f99' : (dd ? '#b4372f' : '#bdbdb7');
        if (dd) ctx.setLineDash([3, 2]);
        ctx.strokeRect(x + 0.5, y + 0.5, T - 1, T - 1);
        ctx.setLineDash([]);
        S.tileRects.push({ x: x, y: y, s: T, col: c2, j: j2 });
      }
    }
    // labels
    ctx.fillStyle = '#8a8a8f';
    ctx.font = '10.5px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var c3 = 0; c3 < cols; c3++) {
      var name = c3 === 0 ? 'input' : c3 === cols - 1 ? 'output' : 'h' + c3;
      ctx.fillText(name, xs[c3], 4);
    }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText('x₁', xs[0] - T / 2 - 4, ys[0][0]);
    ctx.fillText('x₂', xs[0] - T / 2 - 4, ys[0][1]);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('ŷ', xs[cols - 1], ys[cols - 1][0] + T / 2 + 2);
  }

  /* ---------- Rendering: plots, readouts ---------- */
  function renderLegends() {
    var lossItems = [
      '<span class="lg"><i class="line-key" style="border-color:#1f5f99"></i>train</span>',
      '<span class="lg"><i class="line-key dashed" style="border-color:#d9772e"></i>test</span>'
    ];
    if (S.compareSGD) lossItems.push('<span class="lg"><i class="line-key" style="border-color:#9a9a9f"></i>train, plain SGD (same init &amp; batches)</span>');
    el.lossLegend.innerHTML = lossItems.join('');
    var g = [];
    for (var l = 0; l < S.net.L; l++) {
      g.push('<span class="lg"><i class="line-key" style="border-color:' + LAYER_COLORS[l % LAYER_COLORS.length] + '"></i>W<sup>(' + (l + 1) + ')</sup>' + (l === S.net.L - 1 ? ' (output)' : '') + '</span>');
    }
    el.gradLegend.innerHTML = g.join('');
  }

  function drawPlots() {
    var series = [
      { data: S.hist.trL, color: '#1f5f99', width: 1.8 },
      { data: S.hist.teL, color: '#d9772e', width: 1.6, dash: [5, 3] }
    ];
    if (S.compareSGD) series.unshift({ data: S.hist.cmpL, color: '#9a9a9f', width: 1.4, alpha: 0.9 });
    Plots.drawLinePlot(el.lossPlot, series, { logY: S.logLoss });
    var gs = S.hist.grad.map(function (d, l) { return { data: d, color: LAYER_COLORS[l % LAYER_COLORS.length], width: 1.5 }; });
    Plots.drawLinePlot(el.gradPlot, gs, { logY: true });
  }

  function renderReadouts() {
    el.rEpoch.textContent = String(S.epoch);
    el.rTrainLoss.textContent = fmtNum(S.last.trL);
    el.rTestLoss.textContent = fmtNum(S.last.teL);
    el.rTrainAcc.textContent = fmtPct(S.last.trA);
    el.rTestAcc.textContent = fmtPct(S.last.teA);
    var deadCount = 0, hiddenCount = 0;
    if (S.dead) S.dead.forEach(function (r) { r.forEach(function (d) { if (d) deadCount++; }); });
    S.hidden.forEach(function (n) { hiddenCount += n; });
    var parts = [
      N_POINTS + ' points: ' + S.split.train.length + ' train / ' + S.split.test.length + ' test',
      'batch ' + batchSize(),
      S.net.paramCount() + ' parameters'
    ];
    if (S.activation === 'relu') parts.push('dead ReLUs: ' + deadCount + ' / ' + hiddenCount);
    if (S.optimizer === 'adam') parts.push('Adam step t = ' + S.opt.t);
    el.meta.textContent = parts.join(' · ');
    el.netInfo.textContent = S.net.sizes.join('–') + ', ' + NN.ACTIVATIONS[S.activation].label;
  }

  function render() {
    computeMaps();
    drawBoundary();
    drawNetwork();
    drawPlots();
    renderReadouts();
  }

  /* ---------- Main loop ---------- */
  function frame() {
    if (S.playing && !S.diverged) {
      S.acc += SPEED_VALUES[S.speed];
      var k = Math.floor(S.acc);
      S.acc -= k;
      var t0 = performance.now();
      for (var i = 0; i < k; i++) {
        if (!trainOneEpoch()) break;
        if (performance.now() - t0 > FRAME_BUDGET_MS) { S.acc = 0; break; }
      }
    }
    if (S.dirty) { S.dirty = false; render(); }
    requestAnimationFrame(frame);
  }

  function setPlaying(on) {
    S.playing = on && !S.diverged;
    el.play.textContent = S.playing ? 'Pause' : 'Play';
    el.play.setAttribute('aria-label', S.playing ? 'Pause' : 'Play');
  }

  /* ---------- Equations ---------- */
  function tex(target, src, display) {
    if (!target) return;
    if (window.katex) {
      try { window.katex.render(src, target, { displayMode: !!display, throwOnError: false }); return; }
      catch (e) { /* fall through */ }
    }
    target.textContent = src;
    target.classList.add('tex-fallback');
  }
  function renderStaticTex() {
    document.querySelectorAll('[data-tex]').forEach(function (n) {
      tex(n, n.getAttribute('data-tex'), n.classList.contains('tex-block'));
    });
  }
  function renderDynamicTex() {
    var A = {
      tanh: '\\sigma(z) = \\tanh z, \\qquad \\sigma\'(z) = 1 - \\tanh^2 z',
      relu: '\\sigma(z) = \\max(0, z), \\qquad \\sigma\'(z) = \\mathbb{1}[z > 0]',
      sigmoid: '\\sigma(z) = \\frac{1}{1+e^{-z}}, \\qquad \\sigma\'(z) = \\sigma(z)\\,\\big(1-\\sigma(z)\\big)',
      leaky_relu: '\\sigma(z) = \\max(\\alpha z, z), \\qquad \\sigma\'(z) = \\begin{cases} 1 & z > 0 \\\\ \\alpha & z \\le 0 \\end{cases}, \\qquad \\alpha = 0.01'
    };
    tex(el.eqAct, A[S.activation], true);
    var lrS = String(+S.lr.toPrecision(2)).replace(/e-?(\d+)/, function (m) { return m; });
    var eta = '\\eta = ' + (S.lr < 0.01 ? S.lr.toExponential(1).replace(/e(-?\d+)/, '\\times 10^{$1}') : lrS);
    var g = 'g_t = \\nabla_\\theta J_{B_t}(\\theta_{t-1})';
    var blocks;
    if (S.optimizer === 'sgd') {
      el.optName.textContent = 'SGD';
      blocks = [g, '\\theta_t = \\theta_{t-1} - \\eta\\, g_t, \\qquad ' + eta];
    } else if (S.optimizer === 'momentum') {
      el.optName.textContent = 'SGD with momentum';
      blocks = [g, 'v_t = \\mu\\, v_{t-1} + g_t, \\qquad \\theta_t = \\theta_{t-1} - \\eta\\, v_t, \\qquad v_0 = 0', '\\mu = 0.9, \\qquad ' + eta];
    } else {
      el.optName.textContent = 'Adam';
      blocks = [g,
        'm_t = \\beta_1 m_{t-1} + (1-\\beta_1)\\, g_t, \\qquad v_t = \\beta_2 v_{t-1} + (1-\\beta_2)\\, g_t \\odot g_t',
        '\\hat m_t = \\frac{m_t}{1-\\beta_1^{\\,t}}, \\qquad \\hat v_t = \\frac{v_t}{1-\\beta_2^{\\,t}} \\qquad \\text{(bias correction, } m_0 = v_0 = 0)',
        '\\theta_t = \\theta_{t-1} - \\eta\\, \\frac{\\hat m_t}{\\sqrt{\\hat v_t} + \\epsilon}',
        '\\beta_1 = 0.9,\\quad \\beta_2 = 0.999,\\quad \\epsilon = 10^{-8},\\quad ' + eta];
    }
    el.eqOpt.innerHTML = '';
    blocks.forEach(function (b) { var d = document.createElement('div'); d.className = 'tex-block'; el.eqOpt.appendChild(d); tex(d, b, true); });
    var he = S.activation === 'relu' || S.activation === 'leaky_relu';
    tex(el.eqInit, (he
      ? 'W^{(l)}_{jk} \\sim \\mathcal{N}\\!\\left(0, \\tfrac{2}{n_{l-1}}\\right)\\ \\text{(He, hidden layers)}, \\quad '
      : 'W^{(l)}_{jk} \\sim \\mathcal{N}\\!\\left(0, \\tfrac{2}{n_{l-1}+n_l}\\right)\\ \\text{(Xavier/Glorot)}, \\quad ') +
      (he ? 'W^{(L)} \\text{ Xavier}, \\quad ' : '') + 'b^{(l)} = 0', true);
  }

  /* ---------- Controls ---------- */
  function buildLayerSteppers() {
    el.layerSteppers.innerHTML = '';
    S.hidden.forEach(function (n, l) {
      var row = document.createElement('div');
      row.className = 'layer-row';
      var pips = '';
      for (var i = 0; i < n; i++) pips += '<i class="pip"></i>';
      row.innerHTML = '<span class="lname">Layer ' + (l + 1) + '</span>' +
        '<button type="button" class="btn btn-icon" data-l="' + l + '" data-d="-1" aria-label="Fewer neurons in layer ' + (l + 1) + '"' + (n <= 1 ? ' disabled' : '') + '>−</button>' +
        '<span class="count">' + n + '</span>' +
        '<button type="button" class="btn btn-icon" data-l="' + l + '" data-d="1" aria-label="More neurons in layer ' + (l + 1) + '"' + (n >= MAX_NEURONS ? ' disabled' : '') + '>+</button>' +
        '<span class="pips">' + pips + '</span>';
      el.layerSteppers.appendChild(row);
    });
    el.layersOut.textContent = String(S.hidden.length);
    el.layersDec.disabled = S.hidden.length <= 1;
    el.layersInc.disabled = S.hidden.length >= MAX_LAYERS;
  }

  function syncControls() {
    el.dataset.value = S.dataset;
    el.noise.value = Math.round(S.noise * 100); el.noiseOut.textContent = Math.round(S.noise * 100) + '%';
    el.split.value = Math.round(S.trainFrac * 100); el.splitOut.textContent = Math.round(S.trainFrac * 100) + '%';
    el.activation.value = S.activation;
    el.optimizer.value = S.optimizer;
    el.compareSGD.checked = S.compareSGD;
    el.lr.value = lrToSlider(S.lr); el.lrOut.textContent = fmtLr(S.lr);
    var li = nearestIndex(L2_VALUES, S.l2); el.l2.value = li; el.l2Out.textContent = S.l2 === 0 ? '0' : String(L2_VALUES[li]);
    var bi = BATCH_VALUES.indexOf(S.batch); if (bi < 0) bi = nearestIndex(BATCH_VALUES, S.batch);
    el.batch.value = bi; el.batchOut.textContent = S.batch > 0 ? String(S.batch) : 'full';
    el.speed.value = S.speed; el.speedOut.textContent = speedLabel(SPEED_VALUES[S.speed]);
    buildLayerSteppers();
  }
  function speedLabel(v) { return v < 1 ? '1 epoch / ' + Math.round(1 / v) + ' frames' : v + (v === 1 ? ' epoch / frame' : ' epochs / frame'); }

  function clearPreset(note) {
    var was = S.presetId;
    S.presetId = null;
    el.presetList.querySelectorAll('.preset').forEach(function (b) { b.classList.remove('active'); });
    if (was || note || S.watchMode === 'tried') showNeutralWatch(note);
  }

  /* ---------- "What to watch" callout ---------- */
  function presetById(id) { for (var i = 0; i < PRESETS.length; i++) if (PRESETS[i].id === id) return PRESETS[i]; return null; }
  function setWatchClass(kind) {
    el.watch.className = 'watch' + (kind ? ' ' + kind : '');
  }
  function showWatch(p) {
    var w = p.watch;
    if (!w) { el.watch.hidden = true; return; }
    S.watchDismissed = false;
    S.watchMode = 'preset';
    setWatchClass(w.tag ? 'fails' : '');
    el.watchTag.hidden = !w.tag;
    el.watchTag.textContent = w.tag || '';
    el.watchPreset.textContent = p.name;
    el.watchText.textContent = w.text;
    el.watchTry.innerHTML = '';
    var lab = document.createElement('span');
    lab.className = 'try-label'; lab.textContent = 'Try:';
    el.watchTry.appendChild(lab);
    var txt = document.createElement('span');
    txt.textContent = w.tryText;
    el.watchTry.appendChild(txt);
    (w.tries || []).forEach(function (t, i) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'btn'; b.textContent = t.label;
      b.dataset.tryIndex = String(i);
      b.addEventListener('click', function () { applyTry(p, t); });
      el.watchTry.appendChild(b);
    });
    el.watch.hidden = false;
  }
  // Settings no longer match a preset: replace the preset-specific note with a neutral hint.
  function showNeutralWatch(note) {
    if (S.watchDismissed) return;
    S.watchMode = note ? 'tried' : 'neutral';
    setWatchClass('neutral');
    el.watchTag.hidden = true; el.watchTag.textContent = '';
    el.watchPreset.textContent = '';
    el.watchText.textContent = note || 'Settings changed by hand, so the preset note no longer applies. Pick a preset to see what to watch.';
    el.watchTry.innerHTML = '';
    el.watch.hidden = false;
  }
  function applyTry(p, t) {
    if (t.preset) { var q = presetById(t.preset); if (q) applyPreset(q); return; }
    if (t.reset) {
      // keep the preset (and its note); draw a fresh random initialization
      S.initSeed = (Math.random() * 0x7fffffff) | 0; resetTraining(); setPlaying(true); return;
    }
    if (t.set) {
      for (var k in t.set) S[k] = t.set[k];
      syncControls();
      clearPreset('Applied "' + t.label + '" to the ' + p.name + ' setup. Compare the curves with before; click ' + p.name + ' to restore the preset.');
      resetTraining();
      renderDynamicTex();
      setPlaying(true);
    }
  }

  function applyPreset(p) {
    var c = p.config;
    S.dataset = c.dataset; S.noise = c.noise; S.trainFrac = c.trainFrac;
    S.hidden = c.hidden.slice(); S.activation = c.activation; S.optimizer = c.optimizer;
    S.lr = c.lr; S.l2 = c.l2; S.batch = c.batch; S.dataSeed = c.dataSeed; S.initSeed = c.initSeed;
    S.compareSGD = !!c.compareSGD;
    S.speed = nearestIndex(SPEED_VALUES, c.speed);
    S.presetId = p.id;
    el.presetList.querySelectorAll('.preset').forEach(function (b) { b.classList.toggle('active', b.dataset.id === p.id); });
    el.presetDesc.textContent = p.desc;
    showWatch(p);
    syncControls();
    buildData();
    resetTraining();
    renderDynamicTex();
    setPlaying(true);
  }

  function bind() {
    // presets
    PRESETS.forEach(function (p) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'preset'; b.textContent = p.name; b.dataset.id = p.id; b.title = p.desc;
      b.addEventListener('click', function () { applyPreset(p); });
      el.presetList.appendChild(b);
    });
    Object.keys(DS.DATASETS).forEach(function (k) {
      var o = document.createElement('option'); o.value = k; o.textContent = DS.DATASETS[k].label; el.dataset.appendChild(o);
    });

    var dataChanged = function () { clearPreset(); buildData(); resetTraining(); };
    el.dataset.addEventListener('change', function () { S.dataset = el.dataset.value; dataChanged(); });
    el.noise.addEventListener('input', function () { S.noise = +el.noise.value / 100; el.noiseOut.textContent = el.noise.value + '%'; dataChanged(); });
    el.split.addEventListener('input', function () { S.trainFrac = +el.split.value / 100; el.splitOut.textContent = el.split.value + '%'; dataChanged(); });
    el.regen.addEventListener('click', function () {
      S.dataSeed = (Math.random() * 0x7fffffff) | 0; S.initSeed = (Math.random() * 0x7fffffff) | 0; dataChanged();
    });

    var archChanged = function () { clearPreset(); buildLayerSteppers(); resetTraining(); };
    el.layersDec.addEventListener('click', function () { if (S.hidden.length > 1) { S.hidden.pop(); archChanged(); } });
    el.layersInc.addEventListener('click', function () {
      if (S.hidden.length < MAX_LAYERS) { S.hidden.push(S.hidden[S.hidden.length - 1] || 4); archChanged(); }
    });
    el.layerSteppers.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-l]'); if (!b) return;
      var l = +b.dataset.l, d = +b.dataset.d, n = S.hidden[l] + d;
      if (n >= 1 && n <= MAX_NEURONS) { S.hidden[l] = n; archChanged(); }
    });
    el.activation.addEventListener('change', function () { S.activation = el.activation.value; clearPreset(); resetTraining(); renderDynamicTex(); });
    el.optimizer.addEventListener('change', function () {
      // switching optimizer keeps the weights and starts a fresh optimizer state
      S.optimizer = el.optimizer.value; S.opt = new NN.Optimizer(S.net, S.optimizer); clearPreset(); renderDynamicTex(); S.dirty = true;
    });
    el.compareSGD.addEventListener('change', function () { S.compareSGD = el.compareSGD.checked; clearPreset(); resetTraining(); });
    el.lr.addEventListener('input', function () { S.lr = sliderToLr(+el.lr.value); el.lrOut.textContent = fmtLr(S.lr); clearPreset(); renderDynamicTex(); });
    el.l2.addEventListener('input', function () { S.l2 = L2_VALUES[+el.l2.value]; el.l2Out.textContent = String(S.l2); clearPreset(); });
    el.batch.addEventListener('input', function () {
      S.batch = BATCH_VALUES[+el.batch.value]; el.batchOut.textContent = S.batch > 0 ? String(S.batch) : 'full'; clearPreset(); S.dirty = true;
    });
    el.watchClose.addEventListener('click', function () { S.watchDismissed = true; el.watch.hidden = true; });
    el.speed.addEventListener('input', function () { S.speed = +el.speed.value; el.speedOut.textContent = speedLabel(SPEED_VALUES[S.speed]); });

    el.play.addEventListener('click', function () { setPlaying(!S.playing); });
    el.step.addEventListener('click', function () { setPlaying(false); trainOneEpoch(); });
    el.reset.addEventListener('click', function () { S.initSeed = (Math.random() * 0x7fffffff) | 0; resetTraining(); });
    el.showTest.addEventListener('change', function () { S.showTest = el.showTest.checked; S.dirty = true; });
    el.discretize.addEventListener('change', function () { S.discretize = el.discretize.checked; S.dirty = true; });
    el.logLoss.addEventListener('change', function () { S.logLoss = el.logLoss.checked; S.dirty = true; });

    // hover a unit in the diagram to show its activation on the main canvas
    var hit = function (e) {
      var r = el.network.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      for (var i = 0; i < S.tileRects.length; i++) {
        var t = S.tileRects[i];
        if (x >= t.x && x <= t.x + t.s && y >= t.y && y <= t.y + t.s) return t.col === S.net.L ? null : { col: t.col, j: t.j };
      }
      return null;
    };
    el.network.addEventListener('mousemove', function (e) {
      var h = hit(e), prev = S.hover;
      if ((h && (!prev || prev.col !== h.col || prev.j !== h.j)) || (!h && prev)) { S.hover = h; S.dirty = true; }
      el.network.style.cursor = h ? 'pointer' : 'default';
    });
    el.network.addEventListener('mouseleave', function () { if (S.hover) { S.hover = null; S.dirty = true; } });

    document.addEventListener('keydown', function (e) {
      if (e.code !== 'Space') return;
      var t = e.target.tagName;
      if (t === 'INPUT' || t === 'SELECT' || t === 'BUTTON' || t === 'TEXTAREA') return;
      e.preventDefault(); setPlaying(!S.playing);
    });

    var markDirty = function () { S.dirty = true; };
    window.addEventListener('resize', markDirty);
    if (window.ResizeObserver) {
      var ro = new ResizeObserver(markDirty);
      [el.boundary, el.network, el.lossPlot, el.gradPlot].forEach(function (c) { ro.observe(c); });
    }
  }

  /* ---------- Boot ---------- */
  function boot() {
    bind();
    var first = PRESETS[0];
    // Start on the first preset, paused, so the page opens on a meaningful setup.
    applyPreset(first);
    setPlaying(false);
    renderStaticTex();
    requestAnimationFrame(frame);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
