/*
 * datasets.js — seeded 2-D binary classification datasets.
 * All datasets live roughly in [-1, 1]^2. `noise` is in [0, 1].
 */
(function (root, factory) {
  var NNmod = (typeof module === 'object' && module.exports) ? require('./nn.js') : root.NN;
  var mod = factory(NNmod);
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.Datasets = mod;
})(typeof self !== 'undefined' ? self : this, function (NN) {
  'use strict';

  var DATASETS = {
    spiral:  { label: 'Spiral' },
    xor:     { label: 'XOR' },
    circles: { label: 'Circles' },
    moons:   { label: 'Two moons' },
    blobs:   { label: 'Gaussian blobs' }
  };

  function generate(name, n, noise, seed) {
    var rng = NN.mulberry32(seed);
    var g = function () { return NN.gaussian(rng); };
    var X = new Float64Array(2 * n), Y = new Uint8Array(n);
    var half = Math.floor(n / 2);
    var k, c, x, y, t, r, th, s;
    for (k = 0; k < n; k++) {
      c = k < half ? 0 : 1;
      var i = c === 0 ? k : k - half;
      var nc = c === 0 ? half : n - half;
      switch (name) {
        case 'spiral':
          t = i / nc;
          r = 0.06 + 0.94 * t;
          th = 1.75 * 2 * Math.PI * t + c * Math.PI;
          s = 0.1 * noise;
          x = r * Math.sin(th) + s * g();
          y = r * Math.cos(th) + s * g();
          break;
        case 'xor':
          // four Gaussian clusters at (±0.5, ±0.5); class 1 where the signs agree
          var sx = rng() < 0.5 ? -1 : 1;
          var sy = c === 1 ? sx : -sx;
          s = 0.08 + 0.25 * noise;
          x = 0.5 * sx + s * g();
          y = 0.5 * sy + s * g();
          break;
        case 'circles':
          r = c === 1 ? 0.45 * Math.sqrt(rng()) : 0.65 + 0.35 * rng();
          th = 2 * Math.PI * rng();
          s = 0.2 * noise;
          x = r * Math.cos(th) + s * g();
          y = r * Math.sin(th) + s * g();
          break;
        case 'moons':
          t = Math.PI * rng();
          if (c === 0) { x = Math.cos(t); y = Math.sin(t); }
          else { x = 1 - Math.cos(t); y = 0.5 - Math.sin(t); }
          s = 0.3 * noise;
          x = (x - 0.5) / 1.5 + s * g();
          y = (y - 0.25) / 1.5 + s * g();
          break;
        case 'blobs':
          s = 0.15 + 0.35 * noise;
          x = (c === 0 ? -0.45 : 0.45) + s * g();
          y = (c === 0 ? -0.45 : 0.45) + s * g();
          break;
        default:
          throw new Error('Unknown dataset ' + name);
      }
      X[2 * k] = x; X[2 * k + 1] = y; Y[k] = c;
    }
    return { name: name, n: n, X: X, Y: Y };
  }

  // Random (seeded) train/test split; returns index arrays.
  function split(n, trainFrac, seed) {
    var idx = [];
    for (var k = 0; k < n; k++) idx.push(k);
    NN.shuffleInPlace(idx, NN.mulberry32(seed ^ 0x9e3779b9));
    var nTrain = Math.max(2, Math.min(n - 1, Math.round(n * trainFrac)));
    return { train: idx.slice(0, nTrain), test: idx.slice(nTrain) };
  }

  return { DATASETS: DATASETS, generate: generate, split: split, DOMAIN: 1.6 };
});
