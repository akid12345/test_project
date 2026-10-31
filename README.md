# Neural Net Playground

A small multilayer perceptron that trains live in the browser on 2-D binary classification data. The forward pass, backpropagation, L2 regularization and the optimizers (SGD, SGD + momentum, Adam) are written from scratch in plain JavaScript in float64. There are no frameworks and no build step.

It will be hosted at **https://akid12345.github.io/test_project/** once deployed.

## Run locally

Open `index.html` in any modern browser. It needs no server. Equations render with KaTeX from the jsDelivr CDN (pinned to 0.16.11). Offline, the TeX source is shown instead and everything else still works.

## Features

- **Datasets:** 2-arm spiral, XOR (four clusters), concentric circles, two moons and Gaussian blobs. There are sliders for noise and the train/test split, plus a regenerate button. Test points are drawn hollow.
- **Network:** 1–4 hidden layers with 1–8 neurons each. Activations are tanh, ReLU, sigmoid and leaky ReLU. The output is a single sigmoid unit trained with binary cross-entropy. Weights use Xavier init for tanh/sigmoid and He init for the ReLU family.
- **Training:** learning rate on a log scale from 1e-4 to 3, L2 strength, batch size from 1 to full batch, and SGD, momentum or Adam. An optional plain-SGD run starts from the same initial weights and uses the same minibatches, for comparison.
- **Visuals:**
  - Output probability heatmap. It interpolates the logit on a 61×61 grid at display resolution and draws the p = 0.5 decision boundary as a crisp contour line.
  - Network diagram. Edge color shows weight sign and width shows magnitude. Each unit has a thumbnail of its activation over the input plane; hover one to enlarge it on the main canvas. Dead ReLUs are dimmed.
  - Train and test loss curves, with an optional log axis.
  - Per-layer gradient-norm curves.
  - Long training histories are drawn per pixel column: a mean line plus a faint min–max band.
- **What-to-watch notes:** every preset shows a short note above the canvas. It says what to look for, flags presets that fail on purpose, and offers one-click "Try" actions. The note switches to a neutral hint when settings are changed by hand.
- **Readouts:** epoch, train and test loss, train and test accuracy, parameter count, and dead-ReLU count.
- **Controls:** play/pause (Space also works), step one epoch, reset weights, and speed from 1/8 to 25 epochs per frame.
- **Presets:** Spiral needs depth, Too shallow, XOR minimum, Learning rate too high, Overfitting, Regularization fixes it, Dying ReLU, and Adam vs SGD.
- **Equations and explainer:** KaTeX renders the forward pass, the loss, backprop and the update rule for whichever optimizer is selected. A short explainer covers what the network learns, why depth helps, and what the learning rate and regularization do.
- **Privacy:** no accounts, no tracking, and nothing is stored (no cookies or localStorage).

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page structure, equations, explainer |
| `styles.css` | Layout and theme (light, responsive) |
| `nn.js` | MLP, backprop, optimizers (browser global `NN` or Node `require`) |
| `datasets.js` | Seeded dataset generators and train/test split |
| `presets.js` | Preset scenarios (fixed seeds, reproducible) |
| `plots.js` | HiDPI canvas helpers and line plots |
| `app.js` | UI state, training loop, rendering |
| `tests/run_tests.js` | Node tests: gradient check and preset outcomes (repo/dev only; not part of the deployed page) |

## Tests

```
node tests/run_tests.js
```

The test file runs three groups of checks:

- **Gradient check:** central finite differences on a 2-4-3-3-1 network with L2 for every activation. The relative error must be below 1e-5.
- **Adam step:** confirms the bias-corrected first step.
- **Presets:** runs each preset headless, for example Spiral needs depth reaching >95% train accuracy and XOR minimum reaching 100%.

## Notes

- Displayed losses are the mean BCE data term. The L2 penalty is excluded from the readouts but included in the gradients.
- With Adam, L2 is the coupled (gradient) form, not AdamW's decoupled weight decay.
- The heatmap evaluates the network on a 61×61 grid of points and interpolates the logit between them, so it is a close visual approximation. A very thin feature smaller than one grid cell can be missed, so a single point can look slightly misclassified relative to the reported accuracy, which is computed exactly on each data point.
- Several presets use a small L2 term (λ = 1e-4 to 1e-3). Without it, on separable data the cross-entropy keeps pushing the weights larger forever. The probabilities saturate, the heatmap and unit tiles turn into hard blocks, and Adam's late updates make the loss spike. Set λ to 0 to watch that happen.
