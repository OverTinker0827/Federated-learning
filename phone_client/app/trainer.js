/*
 * DemandNet trainer - runs entirely on the phone, no libraries.
 *
 * Mirrors server/model.py exactly:
 *   fc1: Linear(16, 64) -> ReLU -> fc2: Linear(64, 32) -> ReLU -> fc3: Linear(32, 1)
 * Weights use PyTorch's layout: "<layer>.weight" is [out, in] row-major,
 * "<layer>.bias" is [out]. Loss is MSE, optimiser is Adam with PyTorch's
 * update rule, re-created every round just like clientN/train_client.py.
 */
(function (root) {
  'use strict';

  const LAYERS = ['fc1', 'fc2', 'fc3'];

  class DemandNet {
    constructor(sizes = [16, 64, 32, 1]) {
      this.sizes = sizes;
      this.layers = [];
      for (let l = 0; l < sizes.length - 1; l++) {
        const nin = sizes[l], nout = sizes[l + 1];
        const bound = 1 / Math.sqrt(nin);
        const W = new Float32Array(nout * nin).map(() => (Math.random() * 2 - 1) * bound);
        const b = new Float32Array(nout).map(() => (Math.random() * 2 - 1) * bound);
        this.layers.push({ name: LAYERS[l], nin, nout, W, b });
      }
      this.resetOptimizer();
    }

    get numParams() {
      return this.layers.reduce((s, L) => s + L.W.length + L.b.length, 0);
    }

    /* ---------- state dict <-> JSON ({name: {shape, data}}) ---------- */
    loadState(state) {
      for (const L of this.layers) {
        const w = state[`${L.name}.weight`], b = state[`${L.name}.bias`];
        if (!w || !b) throw new Error(`missing ${L.name} parameters`);
        if (w.shape[0] !== L.nout || w.shape[1] !== L.nin) {
          throw new Error(`${L.name}.weight shape ${w.shape} != [${L.nout},${L.nin}]`);
        }
        L.W = Float32Array.from(w.data);
        L.b = Float32Array.from(b.data);
      }
    }

    exportState() {
      const out = {};
      for (const L of this.layers) {
        out[`${L.name}.weight`] = { shape: [L.nout, L.nin], data: Array.from(L.W) };
        out[`${L.name}.bias`] = { shape: [L.nout], data: Array.from(L.b) };
      }
      return out;
    }

    resetOptimizer() {
      this.t = 0;
      this.adam = this.layers.map(L => ({
        mW: new Float32Array(L.W.length), vW: new Float32Array(L.W.length),
        mb: new Float32Array(L.b.length), vb: new Float32Array(L.b.length),
      }));
    }

    /* ---------- forward for one sample, keeping activations ---------- */
    _forward(x, acts) {
      let a = x;
      acts[0] = a;
      for (let l = 0; l < this.layers.length; l++) {
        const L = this.layers[l];
        const z = new Float32Array(L.nout);
        for (let o = 0; o < L.nout; o++) {
          let s = L.b[o];
          const row = o * L.nin;
          for (let i = 0; i < L.nin; i++) s += L.W[row + i] * a[i];
          z[o] = (l < this.layers.length - 1) ? (s > 0 ? s : 0) : s;
        }
        a = z;
        acts[l + 1] = a;
      }
      return a[0];
    }

    predict(X) {
      const acts = [];
      return X.map(x => this._forward(x, acts));
    }

    /* ---------- one optimisation step on a mini-batch ---------- */
    trainBatch(Xb, yb, lr) {
      const B = Xb.length;
      const grads = this.layers.map(L => ({ W: new Float32Array(L.W.length), b: new Float32Array(L.b.length) }));
      let loss = 0;
      const acts = [];
      for (let n = 0; n < B; n++) {
        const pred = this._forward(Xb[n], acts);
        const err = pred - yb[n];
        loss += err * err;
        let delta = new Float32Array([2 * err / B]); // dL/d(out)
        for (let l = this.layers.length - 1; l >= 0; l--) {
          const L = this.layers[l], g = grads[l], ain = acts[l];
          for (let o = 0; o < L.nout; o++) {
            const d = delta[o];
            if (d === 0) continue;
            g.b[o] += d;
            const row = o * L.nin;
            for (let i = 0; i < L.nin; i++) g.W[row + i] += d * ain[i];
          }
          if (l > 0) {
            const prev = new Float32Array(L.nin);
            for (let i = 0; i < L.nin; i++) {
              if (ain[i] <= 0) continue; // ReLU derivative
              let s = 0;
              for (let o = 0; o < L.nout; o++) s += L.W[o * L.nin + i] * delta[o];
              prev[i] = s;
            }
            delta = prev;
          }
        }
      }
      this._adamStep(grads, lr);
      return loss / B;
    }

    _adamStep(grads, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
      this.t += 1;
      const bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
      const step = (p, g, m, v) => {
        for (let i = 0; i < p.length; i++) {
          m[i] = b1 * m[i] + (1 - b1) * g[i];
          v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          p[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
        }
      };
      this.layers.forEach((L, l) => {
        const s = this.adam[l];
        step(L.W, grads[l].W, s.mW, s.vW);
        step(L.b, grads[l].b, s.mb, s.vb);
      });
    }

    /* ---------- full epoch, async so the UI keeps rendering ---------- */
    async trainEpoch(X, y, { lr = 3e-3, batchSize = 32, paceMs = 0, onBatch, shouldStop } = {}) {
      const idx = [...Array(X.length).keys()];
      for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      const batches = Math.ceil(X.length / batchSize);
      let total = 0;
      for (let b = 0; b < batches; b++) {
        if (shouldStop && shouldStop()) throw new Error('training cancelled');
        const sel = idx.slice(b * batchSize, (b + 1) * batchSize);
        const loss = this.trainBatch(sel.map(i => X[i]), sel.map(i => y[i]), lr);
        if (!Number.isFinite(loss)) throw new Error('loss became NaN/Inf');
        total += loss;
        if (onBatch) onBatch({ batch: b + 1, batches, loss, avgLoss: total / (b + 1), samples: sel.length });
        await new Promise(r => setTimeout(r, paceMs));
      }
      return total / batches;
    }
  }

  function mse(pred, y) {
    let s = 0;
    for (let i = 0; i < y.length; i++) s += (pred[i] - y[i]) ** 2;
    return s / y.length;
  }

  const api = { DemandNet, mse };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FLTrainer = api;
})(typeof self !== 'undefined' ? self : this);
