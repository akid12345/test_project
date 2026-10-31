/*
 * nn.js — a small multilayer perceptron for 2-D binary classification.
 * Forward pass, backpropagation, L2 regularization and optimizers (SGD,
 * SGD+momentum, Adam) implemented from scratch in float64.
 * Works in the browser (window.NN) and in Node (module.exports).
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.NN = mod;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ---------- Seeded RNG ---------- */
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rng) {
    var u = 0, v = 0;
    while (u === 0) u = rng();
    v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function shuffleInPlace(arr, rng) {
    for (var i = arr.length - 1; i > 0; i--) {
      var j = Math.floor(rng() * (i + 1));
      var t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  /* ---------- Activations ---------- */
  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    var e = Math.exp(z);
    return e / (1 + e);
  }
  var LEAK = 0.01;
  // df(z, a) receives both pre-activation z and activation a = f(z).
  var ACTIVATIONS = {
    tanh:       { f: Math.tanh, df: function (z, a) { return 1 - a * a; }, init: 'xavier', label: 'tanh' },
    sigmoid:    { f: sigmoid,   df: function (z, a) { return a * (1 - a); }, init: 'xavier', label: 'sigmoid' },
    relu:       { f: function (z) { return z > 0 ? z : 0; }, df: function (z) { return z > 0 ? 1 : 0; }, init: 'he', label: 'ReLU' },
    leaky_relu: { f: function (z) { return z > 0 ? z : LEAK * z; }, df: function (z) { return z > 0 ? 1 : LEAK; }, init: 'he', label: 'leaky ReLU' }
  };

  // Numerically stable binary cross-entropy computed from the logit z.
  // -[y log s(z) + (1-y) log(1-s(z))] = max(z,0) - z*y + log(1+exp(-|z|))
  function bceFromLogit(z, y) {
    return Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
  }

  /* ---------- MLP ---------- */
  function MLP(opts) {
    opts = opts || {};
    var hidden = opts.hidden || [4, 4];
    this.activation = opts.activation || 'tanh';
    this.act = ACTIVATIONS[this.activation];
    if (!this.act) throw new Error('Unknown activation ' + this.activation);
    this.sizes = [opts.inputs || 2].concat(hidden, [1]);
    this.L = this.sizes.length - 1;          // number of weight layers
    this.W = []; this.b = []; this.gW = []; this.gb = [];
    this.zs = []; this.as = []; this.deltas = [];
    this.as.push(new Float64Array(this.sizes[0]));
    for (var l = 0; l < this.L; l++) {
      var nIn = this.sizes[l], nOut = this.sizes[l + 1];
      this.W.push(new Float64Array(nOut * nIn));   // row-major: W[j*nIn + i]
      this.b.push(new Float64Array(nOut));
      this.gW.push(new Float64Array(nOut * nIn));
      this.gb.push(new Float64Array(nOut));
      this.zs.push(new Float64Array(nOut));
      this.as.push(new Float64Array(nOut));
      this.deltas.push(new Float64Array(nOut));
    }
    this.init(opts.seed == null ? 1 : opts.seed);
  }

  // Xavier/Glorot normal for tanh/sigmoid layers and the sigmoid output;
  // He/Kaiming normal for ReLU-family hidden layers. Biases start at 0.
  MLP.prototype.init = function (seed) {
    var rng = mulberry32(seed);
    this.seed = seed;
    for (var l = 0; l < this.L; l++) {
      var nIn = this.sizes[l], nOut = this.sizes[l + 1];
      var isOutput = l === this.L - 1;
      var scheme = isOutput ? 'xavier' : this.act.init;
      var std = scheme === 'he' ? Math.sqrt(2 / nIn) : Math.sqrt(2 / (nIn + nOut));
      var W = this.W[l];
      for (var k = 0; k < W.length; k++) W[k] = std * gaussian(rng);
      this.b[l].fill(0);
    }
  };

  MLP.prototype.paramCount = function () {
    var n = 0;
    for (var l = 0; l < this.L; l++) n += this.W[l].length + this.b[l].length;
    return n;
  };

  MLP.prototype.copyWeightsFrom = function (other) {
    for (var l = 0; l < this.L; l++) { this.W[l].set(other.W[l]); this.b[l].set(other.b[l]); }
  };

  // Forward pass for one input; returns the output logit z^(L).
  MLP.prototype.forward = function (x0, x1) {
    var a0 = this.as[0]; a0[0] = x0; a0[1] = x1;
    var f = this.act.f, L = this.L;
    for (var l = 0; l < L; l++) {
      var nIn = this.sizes[l], nOut = this.sizes[l + 1];
      var W = this.W[l], b = this.b[l], aPrev = this.as[l], z = this.zs[l], a = this.as[l + 1];
      var last = l === L - 1;
      for (var j = 0; j < nOut; j++) {
        var s = b[j], off = j * nIn;
        for (var i = 0; i < nIn; i++) s += W[off + i] * aPrev[i];
        z[j] = s;
        a[j] = last ? sigmoid(s) : f(s);
      }
    }
    return this.zs[L - 1][0];
  };

  MLP.prototype.predict = function (x0, x1) { return sigmoid(this.forward(x0, x1)); };

  // Backward pass for the sample last sent through forward(); accumulates
  // dLoss_i/dW and dLoss_i/db (un-averaged) into gW, gb.
  MLP.prototype.backward = function (y) {
    var L = this.L, df = this.act.df;
    // sigmoid output + BCE: dl/dz^(L) = yhat - y
    this.deltas[L - 1][0] = this.as[L][0] - y;
    for (var l = L - 1; l >= 0; l--) {
      var nIn = this.sizes[l], nOut = this.sizes[l + 1];
      var aPrev = this.as[l], d = this.deltas[l], gW = this.gW[l], gb = this.gb[l];
      for (var j = 0; j < nOut; j++) {
        var dj = d[j], off = j * nIn;
        gb[j] += dj;
        for (var i = 0; i < nIn; i++) gW[off + i] += dj * aPrev[i];
      }
      if (l > 0) {
        // delta^(l-1) = (W^(l))^T delta^(l)  ⊙  sigma'(z^(l-1))
        var W = this.W[l], dPrev = this.deltas[l - 1], zPrev = this.zs[l - 1];
        for (var i2 = 0; i2 < nIn; i2++) {
          var s = 0;
          for (var j2 = 0; j2 < nOut; j2++) s += W[j2 * nIn + i2] * d[j2];
          dPrev[i2] = s * df(zPrev[i2], aPrev[i2]);
        }
      }
    }
  };

  MLP.prototype.zeroGrad = function () {
    for (var l = 0; l < this.L; l++) { this.gW[l].fill(0); this.gb[l].fill(0); }
  };

  MLP.prototype.l2Penalty = function () {
    var s = 0;
    for (var l = 0; l < this.L; l++) { var W = this.W[l]; for (var k = 0; k < W.length; k++) s += W[k] * W[k]; }
    return 0.5 * s;
  };

  // Gradient of J = (1/m) sum_i BCE_i + (lambda/2) sum_l ||W^(l)||_F^2 over
  // samples idx[start..end). Leaves gradients in gW/gb; returns mean BCE.
  MLP.prototype.computeGrads = function (X, Y, idx, start, end, lambda) {
    this.zeroGrad();
    var m = end - start, loss = 0;
    for (var t = start; t < end; t++) {
      var k = idx[t];
      var z = this.forward(X[2 * k], X[2 * k + 1]);
      loss += bceFromLogit(z, Y[k]);
      this.backward(Y[k]);
    }
    var inv = 1 / m;
    for (var l = 0; l < this.L; l++) {
      var gW = this.gW[l], gb = this.gb[l], W = this.W[l];
      for (var q = 0; q < gW.length; q++) gW[q] = gW[q] * inv + lambda * W[q];
      for (var r = 0; r < gb.length; r++) gb[r] *= inv;
    }
    return loss * inv;
  };

  // Total objective (data loss + L2) over idx — used for gradient checking.
  MLP.prototype.objective = function (X, Y, idx, lambda) {
    var loss = 0;
    for (var t = 0; t < idx.length; t++) {
      var k = idx[t];
      loss += bceFromLogit(this.forward(X[2 * k], X[2 * k + 1]), Y[k]);
    }
    return loss / idx.length + lambda * this.l2Penalty();
  };

  // Mean BCE (no penalty) and accuracy over idx.
  MLP.prototype.evaluate = function (X, Y, idx) {
    if (!idx.length) return { loss: NaN, acc: NaN };
    var loss = 0, correct = 0;
    for (var t = 0; t < idx.length; t++) {
      var k = idx[t];
      var z = this.forward(X[2 * k], X[2 * k + 1]);
      loss += bceFromLogit(z, Y[k]);
      if ((z > 0 ? 1 : 0) === Y[k]) correct++;
    }
    return { loss: loss / idx.length, acc: correct / idx.length };
  };

  MLP.prototype.isFinite = function () {
    for (var l = 0; l < this.L; l++) {
      var W = this.W[l];
      for (var k = 0; k < W.length; k++) if (!isFinite(W[k])) return false;
      var b = this.b[l];
      for (var q = 0; q < b.length; q++) if (!isFinite(b[q])) return false;
    }
    return true;
  };

  // Flat list of [array, index] parameter handles (for gradient checks).
  MLP.prototype.paramArrays = function () {
    var out = [];
    for (var l = 0; l < this.L; l++) {
      out.push({ p: this.W[l], g: this.gW[l], name: 'W' + (l + 1) });
      out.push({ p: this.b[l], g: this.gb[l], name: 'b' + (l + 1) });
    }
    return out;
  };

  /* ---------- Optimizers ---------- */
  // sgd:      theta <- theta - eta g
  // momentum: v <- mu v + g;  theta <- theta - eta v
  // adam:     m <- b1 m + (1-b1) g;  v <- b2 v + (1-b2) g^2;
  //           mhat = m/(1-b1^t); vhat = v/(1-b2^t); theta <- theta - eta mhat/(sqrt(vhat)+eps)
  function Optimizer(net, type, opts) {
    opts = opts || {};
    this.type = type || 'adam';
    this.mu = opts.mu != null ? opts.mu : 0.9;
    this.beta1 = opts.beta1 != null ? opts.beta1 : 0.9;
    this.beta2 = opts.beta2 != null ? opts.beta2 : 0.999;
    this.eps = opts.eps != null ? opts.eps : 1e-8;
    this.t = 0;
    this.groups = net.paramArrays().map(function (g) {
      return { p: g.p, g: g.g, m: new Float64Array(g.p.length), v: new Float64Array(g.p.length) };
    });
  }
  Optimizer.prototype.step = function (lr) {
    var type = this.type, G = this.groups;
    if (type === 'sgd') {
      for (var a = 0; a < G.length; a++) {
        var p = G[a].p, g = G[a].g;
        for (var k = 0; k < p.length; k++) p[k] -= lr * g[k];
      }
    } else if (type === 'momentum') {
      var mu = this.mu;
      for (var b = 0; b < G.length; b++) {
        var p2 = G[b].p, g2 = G[b].g, v2 = G[b].v;
        for (var k2 = 0; k2 < p2.length; k2++) { v2[k2] = mu * v2[k2] + g2[k2]; p2[k2] -= lr * v2[k2]; }
      }
    } else if (type === 'adam') {
      this.t++;
      var b1 = this.beta1, b2 = this.beta2, eps = this.eps;
      var c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
      for (var c = 0; c < G.length; c++) {
        var p3 = G[c].p, g3 = G[c].g, m3 = G[c].m, v3 = G[c].v;
        for (var k3 = 0; k3 < p3.length; k3++) {
          var gk = g3[k3];
          m3[k3] = b1 * m3[k3] + (1 - b1) * gk;
          v3[k3] = b2 * v3[k3] + (1 - b2) * gk * gk;
          p3[k3] -= lr * (m3[k3] / c1) / (Math.sqrt(v3[k3] / c2) + eps);
        }
      }
    } else throw new Error('Unknown optimizer ' + type);
  };

  /* ---------- One epoch of minibatch training ---------- */
  // order: a pre-shuffled array of training indices (shared between nets
  // when comparing optimizers). Returns per-layer mean ||dJ/dW^(l)||_F.
  function trainEpoch(net, opt, X, Y, order, batchSize, lr, lambda) {
    var n = order.length, L = net.L;
    var bs = Math.max(1, Math.min(batchSize, n));
    var norms = new Float64Array(L), batches = 0, lossSum = 0;
    for (var s = 0; s < n; s += bs) {
      var e = Math.min(n, s + bs);
      lossSum += net.computeGrads(X, Y, order, s, e, lambda) * (e - s);
      for (var l = 0; l < L; l++) {
        var g = net.gW[l], ss = 0;
        for (var k = 0; k < g.length; k++) ss += g[k] * g[k];
        norms[l] += Math.sqrt(ss);
      }
      opt.step(lr);
      batches++;
    }
    for (var l2 = 0; l2 < L; l2++) norms[l2] /= batches;
    return { gradNorms: Array.prototype.slice.call(norms), meanBatchLoss: lossSum / n };
  }

  // Hidden units whose pre-activation is <= 0 on every given sample (dead ReLUs).
  function deadUnits(net, X, idx) {
    var out = [];
    var L = net.L;
    var alive = [];
    for (var l = 0; l < L - 1; l++) alive.push(new Uint8Array(net.sizes[l + 1]));
    for (var t = 0; t < idx.length; t++) {
      var k = idx[t];
      net.forward(X[2 * k], X[2 * k + 1]);
      for (var l2 = 0; l2 < L - 1; l2++) {
        var z = net.zs[l2], al = alive[l2];
        for (var j = 0; j < z.length; j++) if (z[j] > 0) al[j] = 1;
      }
    }
    for (var l3 = 0; l3 < L - 1; l3++) {
      var row = [];
      for (var j2 = 0; j2 < alive[l3].length; j2++) row.push(!alive[l3][j2]);
      out.push(row);
    }
    return out;
  }

  return {
    MLP: MLP, Optimizer: Optimizer, trainEpoch: trainEpoch, deadUnits: deadUnits,
    ACTIVATIONS: ACTIVATIONS, sigmoid: sigmoid, bceFromLogit: bceFromLogit,
    mulberry32: mulberry32, gaussian: gaussian, shuffleInPlace: shuffleInPlace
  };
});
