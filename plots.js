/*
 * plots.js — HiDPI canvas helpers and a minimal line plot (linear / log y).
 */
(function (root) {
  'use strict';

  // Resize a canvas backing store to its CSS box * devicePixelRatio.
  // Returns {ctx, w, h} in CSS pixels with the transform already applied.
  function fitCanvas(canvas) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    var rect = canvas.getBoundingClientRect();
    var w = Math.max(1, Math.round(rect.width)), h = Math.max(1, Math.round(rect.height));
    var bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h, dpr: dpr };
  }

  function niceStep(span, target) {
    var raw = span / Math.max(1, target);
    var p = Math.pow(10, Math.floor(Math.log10(raw)));
    var m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }
  function fmtTick(v) {
    if (v === 0) return '0';
    var a = Math.abs(v);
    if (a >= 1e4 || a < 1e-2) return v.toExponential(0).replace('e+', 'e');
    return String(+v.toPrecision(3));
  }

  /*
   * series: [{ data: number[], color, width, dash, alpha }]
   * opts: { logY, minLogFloor }
   */
  /*
   * series: [{ data: number[], color, width, dash, alpha }]
   * opts: { logY }
   * Long histories are decimated per pixel column: a faint min-max band keeps
   * spikes visible and a line through the bucket mean (log-mean on a log axis)
   * keeps the curve legible however many epochs have run.
   */
  function drawLinePlot(canvas, series, opts) {
    opts = opts || {};
    var f = fitCanvas(canvas), ctx = f.ctx, W = f.w, H = f.h;
    ctx.save();
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over'; ctx.setLineDash([]);
    ctx.clearRect(0, 0, W, H);
    var padL = 44, padR = 12, padT = 8, padB = 22;
    var pw = W - padL - padR, ph = H - padT - padB;
    var n = 0;
    series.forEach(function (s) { n = Math.max(n, s.data.length); });
    var logY = !!opts.logY;
    var ymin = Infinity, ymax = -Infinity;
    series.forEach(function (s) {
      for (var i = 0; i < s.data.length; i++) {
        var v = s.data[i];
        if (!isFinite(v) || (logY && v <= 0)) continue;
        if (v < ymin) ymin = v;
        if (v > ymax) ymax = v;
      }
    });
    ctx.font = '10.5px ' + getComputedStyle(document.body).fontFamily;
    ctx.fillStyle = '#8a8a8f';
    if (!n || !isFinite(ymin)) {
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('Press Play or Step to start training', W / 2, H / 2);
      ctx.restore();
      return;
    }
    // Values <= 0 cannot be shown on a log axis; they are pinned to the bottom.
    var floorV = ymin;
    var T = logY ? function (v) { return Math.log10(v > 0 ? v : floorV); } : function (v) { return v; };
    var lo = T(ymin), hi = T(ymax);
    if (logY) { lo = Math.floor(lo * 2) / 2; hi = Math.ceil(hi * 2) / 2; if (hi - lo < 1) { hi += 0.5; lo -= 0.5; } }
    else { if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; } lo = Math.min(0, lo); hi = hi + (hi - lo) * 0.05; }
    var xmax = Math.max(n, 2);
    var X = function (i) { return padL + (i / (xmax - 1)) * pw; };
    var Yt = function (t) { return padT + ph - ((t - lo) / (hi - lo)) * ph; };

    // grid + y ticks (decade labels thinned so they never collide)
    ctx.strokeStyle = '#eeeeea'; ctx.lineWidth = 1;
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    if (logY) {
      var e0 = Math.ceil(lo), e1 = Math.floor(hi);
      var every = Math.max(1, Math.ceil((e1 - e0 + 1) / Math.max(2, Math.floor(ph / 18))));
      for (var e = e0; e <= e1; e++) {
        if ((e - e0) % every) continue;
        var yy = Yt(e);
        ctx.beginPath(); ctx.moveTo(padL, Math.round(yy) + 0.5); ctx.lineTo(padL + pw, Math.round(yy) + 0.5); ctx.stroke();
        ctx.fillText('1e' + e, padL - 5, yy);
      }
    } else {
      var st = niceStep(hi - lo, 4);
      for (var t = Math.ceil(lo / st) * st; t <= hi + 1e-12; t += st) {
        var y2 = Yt(t);
        ctx.beginPath(); ctx.moveTo(padL, Math.round(y2) + 0.5); ctx.lineTo(padL + pw, Math.round(y2) + 0.5); ctx.stroke();
        ctx.fillText(fmtTick(t), padL - 5, y2);
      }
    }
    // x ticks (epochs, 1-based); keep the last label inside the canvas
    ctx.textBaseline = 'top';
    var xs = niceStep(n, Math.max(2, Math.floor(pw / 70)));
    for (var xt = xs; xt <= n; xt += xs) {
      var xp = X(xt - 1), lbl = String(xt), tw = ctx.measureText(lbl).width;
      ctx.textAlign = xp + tw / 2 > W - 1 ? 'right' : 'center';
      ctx.fillText(lbl, Math.min(xp, W - 1), padT + ph + 5);
    }
    ctx.strokeStyle = '#cfcfca';
    ctx.beginPath(); ctx.moveTo(padL + 0.5, padT); ctx.lineTo(padL + 0.5, padT + ph); ctx.lineTo(padL + pw, padT + ph + 0.5); ctx.stroke();

    ctx.beginPath(); ctx.rect(padL, padT - 2, pw + 2, ph + 4); ctx.clip();
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    series.forEach(function (s) {
      var d = s.data, len = d.length; if (!len) return;
      var bucket = Math.max(1, Math.ceil(len / pw));
      var xsA = [], mean = [], mn = [], mx = [];
      for (var i = 0; i < len; i += bucket) {
        var end = Math.min(len, i + bucket), sum = 0, cnt = 0, bmin = Infinity, bmax = -Infinity;
        for (var k = i; k < end; k++) {
          var v = d[k]; if (!isFinite(v)) continue;
          var tv = T(v); sum += tv; cnt++;
          if (tv < bmin) bmin = tv; if (tv > bmax) bmax = tv;
        }
        if (!cnt) continue;
        xsA.push(X((i + end - 1) / 2)); mean.push(sum / cnt); mn.push(bmin); mx.push(bmax);
      }
      var a = s.alpha == null ? 1 : s.alpha;
      if (bucket > 1 && xsA.length > 1) {
        ctx.globalAlpha = 0.16 * a; ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.moveTo(xsA[0], Yt(mx[0]));
        for (var q = 1; q < xsA.length; q++) ctx.lineTo(xsA[q], Yt(mx[q]));
        for (var r = xsA.length - 1; r >= 0; r--) ctx.lineTo(xsA[r], Yt(mn[r]));
        ctx.closePath(); ctx.fill();
      }
      ctx.globalAlpha = a; ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6;
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      if (xsA.length === 1) { ctx.moveTo(xsA[0] - 2, Yt(mean[0])); ctx.lineTo(xsA[0] + 2, Yt(mean[0])); }
      for (var m = 0; m < xsA.length; m++) { if (m === 0) ctx.moveTo(xsA[m], Yt(mean[m])); else ctx.lineTo(xsA[m], Yt(mean[m])); }
      ctx.stroke();
      ctx.setLineDash([]);
    });
    ctx.restore();
  }

  root.Plots = { fitCanvas: fitCanvas, drawLinePlot: drawLinePlot };
})(window);
