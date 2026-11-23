import React, { useEffect, useMemo, useState } from 'react';
import { ServerAPI } from '../services/api';
import './ModelBuilder.css';

const ACT_LABEL = { relu: 'ReLU', leaky_relu: 'Leaky ReLU', tanh: 'Tanh', sigmoid: 'Sigmoid', linear: 'Linear' };
const ACT_COLOR = { relu: '#3b82f6', leaky_relu: '#6366f1', tanh: '#10b981', sigmoid: '#f59e0b', linear: '#6b7280' };
const MAX_DRAWN = 10;   // neurons drawn per layer in the diagram

const FALLBACK_PRESET = {
  name: 'Preset 1 - DemandNet',
  input_dim: 16,
  layers: [{ units: 64, activation: 'relu' }, { units: 32, activation: 'relu' }, { units: 1, activation: 'linear' }],
};

const paramCount = (arch) => {
  let prev = arch.input_dim;
  return arch.layers.reduce((sum, l) => { const n = prev * l.units + l.units; prev = l.units; return sum + n; }, 0);
};

function NetworkDiagram({ arch }) {
  const cols = [{ units: arch.input_dim, activation: 'input' }, ...arch.layers];
  const W = 760, H = 300, padX = 50, padY = 28;
  const colX = (i) => padX + (i * (W - 2 * padX)) / Math.max(1, cols.length - 1);
  const nodes = cols.map((c, i) => {
    const shown = Math.min(c.units, MAX_DRAWN);
    const gap = (H - 2 * padY - 20) / Math.max(1, MAX_DRAWN - 1);
    const top = H / 2 - 10 - ((shown - 1) * gap) / 2;
    return Array.from({ length: shown }, (_, k) => ({ x: colX(i), y: top + k * gap }));
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mb-diagram" role="img" aria-label="Network diagram">
      {nodes.slice(1).map((col, i) => col.map((b, bi) => nodes[i].map((a, ai) => (
        <line key={`e${i}-${ai}-${bi}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="mb-edge" />
      ))))}
      {nodes.map((col, i) => {
        const c = cols[i];
        const color = i === 0 ? '#9ca3af' : ACT_COLOR[c.activation];
        return (
          <g key={`c${i}`}>
            {col.map((n, k) => <circle key={k} cx={n.x} cy={n.y} r={7} fill={color} className="mb-node" />)}
            {c.units > MAX_DRAWN && (
              <text x={colX(i)} y={col[col.length - 1].y + 20} className="mb-more">+{c.units - MAX_DRAWN} more</text>
            )}
            <text x={colX(i)} y={H - 18} className="mb-col-title">
              {i === 0 ? 'Input' : i === cols.length - 1 ? 'Output' : `Hidden ${i}`}
            </text>
            <text x={colX(i)} y={H - 5} className="mb-col-sub">
              {c.units}{i === 0 ? ' features' : ` · ${ACT_LABEL[c.activation]}`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ModelBuilder({ serverRunning }) {
  const [arch, setArch] = useState(FALLBACK_PRESET);
  const [saved, setSaved] = useState(null);
  const [presets, setPresets] = useState({ preset1: FALLBACK_PRESET });
  const [activations, setActivations] = useState(Object.keys(ACT_LABEL));
  const [limits, setLimits] = useState({ max_hidden_layers: 6, max_units: 256, input_dim: 16 });
  const [presetKey, setPresetKey] = useState('preset1');
  const [message, setMessage] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    ServerAPI.getModelConfig()
      .then((cfg) => {
        setArch(cfg.arch); setSaved(cfg.arch); setPresets(cfg.presets);
        setActivations(cfg.activations); setLimits(cfg.limits);
      })
      .catch(() => setMessage({ type: 'error', text: 'Could not load the model configuration. Is the server API running?' }));
  }, []);

  const hidden = arch.layers.slice(0, -1);
  const output = arch.layers[arch.layers.length - 1];
  const params = useMemo(() => paramCount(arch), [arch]);
  const dirty = saved && JSON.stringify(saved) !== JSON.stringify(arch);

  const setHidden = (next) => setArch({ ...arch, layers: [...next, output] });
  const updateLayer = (i, patch) => setHidden(hidden.map((l, k) => (k === i ? { ...l, ...patch } : l)));
  const removeLayer = (i) => setHidden(hidden.filter((_, k) => k !== i));
  const moveLayer = (i, d) => {
    const j = i + d;
    if (j < 0 || j >= hidden.length) return;
    const next = [...hidden];
    [next[i], next[j]] = [next[j], next[i]];
    setHidden(next);
  };
  const addLayer = () => {
    const last = hidden[hidden.length - 1];
    setHidden([...hidden, { units: last ? Math.max(1, Math.round(last.units / 2)) : 32, activation: 'relu' }]);
  };
  const clampUnits = (v) => Math.max(1, Math.min(limits.max_units, parseInt(v, 10) || 1));

  const loadPreset = () => {
    setArch(JSON.parse(JSON.stringify(presets[presetKey])));
    setMessage({ type: 'info', text: `${presets[presetKey].name} loaded. Click "Save & apply" to use it for training.` });
  };

  const save = async () => {
    setBusy(true);
    try {
      const res = await ServerAPI.saveModelConfig(arch);
      setSaved(res.arch); setArch(res.arch);
      setMessage({ type: 'success', text: `Saved (${res.num_params.toLocaleString()} parameters). The next training run uses this network on every client, including the phone.` });
    } catch (err) {
      setMessage({ type: 'error', text: err.response?.data?.error || err.message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="model-builder">
      <div className="mb-header">
        <div>
          <h2>Model Builder</h2>
          <p className="mb-sub">Assemble the MLP used for federated training. All clients receive the architecture with the global weights.</p>
        </div>
        <div className="mb-presets">
          <select value={presetKey} onChange={(e) => setPresetKey(e.target.value)}>
            {Object.entries(presets).map(([k, p]) => <option key={k} value={k}>{p.name}</option>)}
          </select>
          <button className="mb-btn" onClick={loadPreset}>Load preset</button>
        </div>
      </div>

      <div className="mb-card">
        <NetworkDiagram arch={arch} />
        <div className="mb-stats">
          <span><b>{hidden.length}</b> hidden layer{hidden.length === 1 ? '' : 's'}</span>
          <span><b>{params.toLocaleString()}</b> parameters</span>
          <span>Fully connected (dense) between consecutive layers</span>
          {dirty && <span className="mb-unsaved">Unsaved changes</span>}
        </div>
      </div>

      <div className="mb-card">
        <div className="mb-name">
          <label>Model name</label>
          <input value={arch.name} maxLength={60} onChange={(e) => setArch({ ...arch, name: e.target.value })} />
        </div>

        <div className="mb-layer mb-fixed">
          <span className="mb-layer-title">Input</span>
          <span>{arch.input_dim} features (listing attributes)</span>
        </div>

        {hidden.map((l, i) => (
          <div className="mb-layer" key={i}>
            <span className="mb-layer-title">Hidden {i + 1}</span>
            <label className="mb-field">Neurons
              <input type="number" min={1} max={limits.max_units} value={l.units}
                onChange={(e) => updateLayer(i, { units: clampUnits(e.target.value) })} />
            </label>
            <input type="range" min={1} max={limits.max_units} value={l.units} className="mb-range"
              onChange={(e) => updateLayer(i, { units: clampUnits(e.target.value) })} />
            <label className="mb-field">Activation
              <select value={l.activation} onChange={(e) => updateLayer(i, { activation: e.target.value })}>
                {activations.map((a) => <option key={a} value={a}>{ACT_LABEL[a] || a}</option>)}
              </select>
            </label>
            <div className="mb-layer-actions">
              <button onClick={() => moveLayer(i, -1)} disabled={i === 0} title="Move up">↑</button>
              <button onClick={() => moveLayer(i, 1)} disabled={i === hidden.length - 1} title="Move down">↓</button>
              <button onClick={() => removeLayer(i)} title="Remove layer">✕</button>
            </div>
          </div>
        ))}

        <button className="mb-btn mb-add" onClick={addLayer} disabled={hidden.length >= limits.max_hidden_layers}>
          + Add hidden layer {hidden.length >= limits.max_hidden_layers ? `(max ${limits.max_hidden_layers})` : ''}
        </button>

        <div className="mb-layer mb-fixed">
          <span className="mb-layer-title">Output</span>
          <span>1 neuron · {ACT_LABEL[output.activation]} (predicted units sold in 7 days)</span>
        </div>
      </div>

      {serverRunning && <div className="mb-msg mb-msg-info">Stop the server to change the architecture.</div>}
      {message && <div className={`mb-msg mb-msg-${message.type}`}>{message.text}</div>}

      <div className="mb-actions">
        <button className="mb-btn" onClick={() => saved && setArch(saved)} disabled={!dirty}>Revert to saved</button>
        <button className="mb-btn mb-primary" onClick={save} disabled={busy || serverRunning || !dirty}>
          {busy ? 'Saving…' : 'Save & apply'}
        </button>
      </div>
    </div>
  );
}

export default ModelBuilder;
