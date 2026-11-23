/*
 * MLP trainer - runs entirely on the phone, no libraries.
 *
 * Mirrors server/model.py: the network is built from the same architecture
 * description the server sends with every round,
 *   {input_dim, layers: [{units, activation}, ...]}   (last layer = 1 linear output)
 * Weights use PyTorch's layout: "fcN.weight" is [out, in] row-major, "fcN.bias" is [out].
 * Loss is MSE, optimiser is Adam with PyTorch's update rule, re-created every round
 * just like clientN/train_client.py.
 */
(function (root) {
  'use strict';

  const PRESET_1 = {
    name: 'Preset 1 - DemandNet',
    input_dim: 16,
    layers: [{ units: 64, activation: 'relu' }, { units: 32, activation: 'relu' }, { units: 1, activation: 'linear' }],
  };

  // f(z) and f'(z) expressed through the activation output a = f(z)
  const ACT = {
    relu: { f: z => (z > 0 ? z : 0), d: a => (a > 0 ? 1 : 0) },
    leaky_relu: { f: z => (z > 0 ? z : 0.01 * z), d: a => (a > 0 ? 1 : 0.01) },
    tanh: { f: z => Math.tanh(z), d: a => 1 - a * a },
    sigmoid: { f: z => 1 / (1 + Math.exp(-z)), d: a => a * (1 - a) },
    linear: { f: z => z, d: () => 1 },
  };

  class DemandNet {
    constructor(arch = PRESET_1) {
      this.arch = arch;
      this.layers = [];
      let nin = arch.input_dim;
      arch.layers.forEach((spec, l) => {
        if (!ACT[spec.activation]) throw new Error(`unknown activation ${spec.activation}`);
        const nout = spec.units;
        const bound = 1 / Math.sqrt(nin);
        const W = new Float32Array(nout * nin).map(() => (Math.random() * 2 - 1) * bound);
        const b = new Float32Array(nout).map(() => (Math.random() * 2 - 1) * bound);
        this.layers.push({ name: `fc${l + 1}`, nin, nout, W, b, act: ACT[spec.activation] });
        nin = nout;
      });
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
          z[o] = L.act.f(s);
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
      const nL = this.layers.length;
      let loss = 0;
      const acts = [];
      for (let n = 0; n < B; n++) {
        const pred = this._forward(Xb[n], acts);
        const err = pred - yb[n];
        loss += err * err;
        // dL/dz of the output layer
        let delta = new Float32Array([(2 * err / B) * this.layers[nL - 1].act.d(pred)]);
        for (let l = nL - 1; l >= 0; l--) {
          const L = this.layers[l], g = grads[l], ain = acts[l];
          for (let o = 0; o < L.nout; o++) {
            const d = delta[o];
            if (d === 0) continue;
            g.b[o] += d;
            const row = o * L.nin;
            for (let i = 0; i < L.nin; i++) g.W[row + i] += d * ain[i];
          }
          if (l > 0) {
            const dPrev = this.layers[l - 1].act.d;
            const prev = new Float32Array(L.nin);
            for (let i = 0; i < L.nin; i++) {
              const di = dPrev(ain[i]);
              if (di === 0) continue;
              let s = 0;
              for (let o = 0; o < L.nout; o++) s += L.W[o * L.nin + i] * delta[o];
              prev[i] = s * di;
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

  const api = { DemandNet, mse, PRESET_1, ACTIVATIONS: Object.keys(ACT) };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FLTrainer = api;
})(typeof self !== 'undefined' ? self : this);
