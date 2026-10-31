/*
 * presets.js — one-click scenarios. Seeds are fixed so each preset is
 * reproducible; "Reset weights" afterwards draws a fresh random init.
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.Presets = mod;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  var BASE = {
    dataset: 'spiral', noise: 0.1, trainFrac: 0.7, n: 400,
    hidden: [6, 6], activation: 'tanh', optimizer: 'adam',
    lr: 0.01, l2: 0, batch: 16, speed: 1, dataSeed: 7, initSeed: 3, compareSGD: false
  };
  function P(id, name, desc, cfg, watch) {
    var c = {};
    for (var k in BASE) c[k] = BASE[k];
    for (var k2 in cfg) c[k2] = cfg[k2];
    return { id: id, name: name, desc: desc, config: c, watch: watch };
  }
  /*
   * watch: the "What to watch" callout shown while a preset is active.
   *   tag:   optional label ('Fails on purpose', 'Partly fails on purpose')
   *   text:  2-3 sentences: what happens and where to look
   *   tryText: the suggested experiment
   *   tries: buttons; { label, preset: id } switches preset,
   *          { label, set: {...} } changes settings and resets the weights.
   */
  var PRESETS = [
    P('spiral-depth', 'Spiral needs depth',
      'Three tanh layers of 8 compose folds of the plane until the two arms separate. A little L2 (λ = 3e-4) keeps the weights bounded once the data are fit.',
      { dataset: 'spiral', noise: 0.1, hidden: [8, 8, 8], activation: 'tanh', optimizer: 'adam', lr: 0.01, l2: 3e-4, batch: 16, speed: 3, initSeed: 6 },
      {
        text: 'Three layers of 8 tanh units. The first layer can only draw straight soft edges; each later layer bends and combines them, until the boundary wraps both arms (train accuracy passes 95% around epoch 200). Watch the h3 tiles in the network diagram turn into spiral pieces.',
        tryText: 'Compare with a one-layer, two-unit net.',
        tries: [{ label: 'Switch to Too shallow', preset: 'too-shallow' }]
      }),
    P('too-shallow', 'Too shallow',
      'One hidden layer with two units can only draw two soft half-planes: the spiral is out of reach.',
      { dataset: 'spiral', noise: 0.1, hidden: [2], activation: 'tanh', optimizer: 'adam', lr: 0.03, l2: 1e-3, batch: 16, speed: 2 },
      {
        tag: 'Fails on purpose',
        text: 'One hidden layer with 2 units gives the output only two roughly straight cuts to combine, which can never wrap a spiral. Train accuracy stalls near 63% and the loss curve flattens around 0.63; the heatmap stays a band or wedge.',
        tryText: 'Add depth and width.',
        tries: [{ label: 'Switch to Spiral needs depth', preset: 'spiral-depth' }]
      }),
    P('xor-min', 'XOR minimum',
      'Two hidden tanh units are the fewest that can represent XOR (one unit yields a single monotone ridge). Tiny nets have bad local minima: try Reset a few times.',
      { dataset: 'xor', noise: 0.0, hidden: [2], activation: 'tanh', optimizer: 'adam', lr: 0.01, l2: 1e-4, batch: 16, speed: 1, initSeed: 1 },
      {
        text: 'Two hidden units are the fewest that can solve XOR: each draws one line, and the output unit keeps the diagonal band between them (watch the two h1 tiles). About 1 in 4 random inits gets stuck near 75% accuracy; that is a genuine local minimum, not a bug.',
        tryText: 'Reset weights a few times and see which inits succeed.',
        tries: [{ label: 'Reset weights', reset: true }]
      }),
    P('lr-high', 'Learning rate too high',
      'SGD+momentum with η = 3 (effective step η/(1−μ) = 30) overshoots every valley: the loss jumps around instead of settling.',
      { dataset: 'circles', noise: 0.1, hidden: [4, 4], activation: 'tanh', optimizer: 'momentum', lr: 3, batch: 32, speed: 1 },
      {
        tag: 'Fails on purpose',
        text: 'With η = 3 and momentum (effective step about 30), every update overshoots the minimum. The loss plot bounces between about 0.7 and 10, accuracy sits at 50%, and the heatmap flips around instead of settling.',
        tryText: 'Use a sensible step size: with η = 0.1 it reaches 100% within about 50 epochs.',
        tries: [{ label: 'Set η = 0.1 and reset', set: { lr: 0.1 } }]
      }),
    P('overfit', 'Overfitting',
      'Noisy data, only 20% for training, a large net and no L2: training loss keeps falling while test loss turns up.',
      { dataset: 'blobs', noise: 0.85, trainFrac: 0.2, hidden: [8, 8, 8], activation: 'tanh', optimizer: 'adam', lr: 0.003, l2: 0, batch: 8, speed: 3, dataSeed: 11 },
      {
        tag: 'Fails on purpose',
        text: 'A large net, only 80 noisy training points and no L2. Train loss falls toward zero while the dashed test loss climbs (from about 0.3 to above 3), and the boundary carves islands around individual noisy points.',
        tryText: 'Add weight decay to the same setup.',
        tries: [{ label: 'Switch to Regularization fixes it', preset: 'l2-fix' }]
      }),
    P('l2-fix', 'Regularization fixes it',
      'Same data and network with L2 weight decay: the boundary stays smooth and test loss stays low.',
      { dataset: 'blobs', noise: 0.85, trainFrac: 0.2, hidden: [8, 8, 8], activation: 'tanh', optimizer: 'adam', lr: 0.003, l2: 0.01, batch: 8, speed: 3, dataSeed: 11 },
      {
        text: 'Same data and network as Overfitting, plus L2 with λ = 0.01. Weights stay small (thin edges in the diagram), the boundary stays smooth, and test loss stays near 0.25 instead of climbing.',
        tryText: 'Compare with the unregularized run.',
        tries: [{ label: 'Switch to Overfitting', preset: 'overfit' }]
      }),
    P('dying-relu', 'Dying ReLU',
      'A large SGD step pushes many ReLU units to negative pre-activations everywhere; their gradient is zero, so they never recover (dimmed in the diagram).',
      { dataset: 'circles', noise: 0.1, hidden: [8, 8, 8], activation: 'relu', optimizer: 'sgd', lr: 1, batch: 8, speed: 1 },
      {
        tag: 'Partly fails on purpose',
        text: 'Large SGD steps (η = 1) push some ReLU units negative for every input; their gradient is then exactly zero, so they never recover. They appear dimmed with a dashed red border in the diagram (about 7 of 24 here), and the net still fits the circles with the units that are left.',
        tryText: 'Use a smaller step, or an activation that never has zero gradient.',
        tries: [{ label: 'Set η = 0.03 and reset', set: { lr: 0.03 } }, { label: 'Switch to leaky ReLU and reset', set: { activation: 'leaky_relu' } }]
      }),
    P('adam-vs-sgd', 'Adam vs SGD',
      'Adam (solid) against plain SGD from the same initial weights and batches (faint line) on two moons.',
      { dataset: 'moons', noise: 0.1, hidden: [8, 8], activation: 'tanh', optimizer: 'adam', lr: 0.01, l2: 1e-4, batch: 16, speed: 1, compareSGD: true },
      {
        text: 'The faint grey line is plain SGD trained from the identical initial weights on the same minibatches. Adam rescales each weight\'s step by a running estimate of its gradient size, so it reaches 97% accuracy in about 20 epochs while SGD needs hundreds. Watch the gap between the two lines on the loss plot.',
        tryText: 'Swap Adam for SGD+momentum: it reaches 97% in about 60 epochs, versus about 530 for plain SGD.',
        tries: [{ label: 'Set optimizer to SGD+momentum and reset', set: { optimizer: 'momentum' } }]
      })
  ];
  return { PRESETS: PRESETS, BASE: BASE };
});
