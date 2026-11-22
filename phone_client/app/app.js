/* EdgeTrain - phone-side federated learning client UI */
(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
  };

  const LR = 3e-3;
  const BATCH = 32;

  const S = {
    base: null,
    data: null,          // {X, y, raw, y_mean, y_std, store}
    net: new FLTrainer.DemandNet(),
    lastRound: 0,
    trainingRound: 0,
    busy: false,
    localPhase: null,
    pollTimer: null,
    failures: 0,
    lossPts: [],
    roundMarks: [],
    batchCounter: 0,
    device: {},
    wakeLock: null,
    finalShown: false,
    state: null,
    bytesDown: 0,
    bytesUp: 0,
    installEvt: null,
  };

  /* ------------------------------------------------------------ helpers */
  function log(msg, level = 'info') {
    const el = $('log');
    const ts = new Date().toLocaleTimeString([], { hour12: false });
    const line = `[${ts}] ${level === 'error' ? '⚠ ' : ''}${msg}\n`;
    el.textContent = (line + el.textContent).split('\n').slice(0, 250).join('\n');
  }
  const fmtBytes = b => b > 1e6 ? `${(b / 1e6).toFixed(2)} MB` : `${(b / 1024).toFixed(1)} KB`;
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  async function api(path, { method = 'GET', body, timeout = 8000 } = {}) {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeout);
    try {
      const res = await fetch(S.base + path, {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? (typeof body === 'string' ? body : JSON.stringify(body)) : undefined,
        signal: ctl.signal,
        cache: 'no-store',
      });
      const text = await res.text();
      let json = {};
      try { json = text ? JSON.parse(text) : {}; } catch { /* non-json */ }
      if (!res.ok) {
        const err = new Error(json.error || `HTTP ${res.status}`);
        err.status = res.status;
        throw err;
      }
      json.__bytes = text.length;
      return json;
    } finally {
      clearTimeout(t);
    }
  }

  /* ------------------------------------------------------------ connect */
  function showView(name) {
    $('view-connect').hidden = name !== 'connect';
    $('view-app').hidden = name !== 'app';
    if (name === 'app') emit('fl:shown');
  }

  const emit = (name, detail) => document.dispatchEvent(new CustomEvent(name, { detail }));

  function initConnectForm() {
    const isLocal = ['localhost', '127.0.0.1', ''].includes(location.hostname);
    $('host').value = store.get('fl.host', isLocal ? '' : location.hostname);
    $('port').value = store.get('fl.port', location.port || '7000');
    $('devname').value = store.get('fl.devname', '');
    $('connect-form').addEventListener('submit', e => {
      e.preventDefault();
      connect($('host').value.trim(), $('port').value.trim());
    });
  }

  async function connect(host, port) {
    const err = $('connect-error');
    err.hidden = true;
    if (!host) return;
    $('connect-btn').disabled = true;
    $('connect-btn').textContent = 'Connecting…';
    S.base = `${location.protocol === 'https:' ? 'https' : 'http'}://${host}:${port}`;
    try {
      const hello = await api('/phone/hello', { timeout: 5000 });
      store.set('fl.host', host);
      store.set('fl.port', port);
      store.set('fl.devname', $('devname').value.trim());
      $('conn-label').textContent = `${host}:${port}`;
      $('round-total').textContent = hello.rounds ?? '–';
      $('epoch-total').textContent = hello.epochs ?? '–';
      log(`Connected to laptop bridge ${host}:${port} (FL server ${hello.server})`);
      await Promise.all([loadDataset(), collectDevice()]);
      showView('app');
      startPolling();
      fetchFinalModel(true);   // a model from an earlier run powers shop forecasts right away
    } catch (e) {
      err.textContent = `Could not reach ${host}:${port} - ${e.name === 'AbortError' ? 'timed out' : e.message}. ` +
        'Is phone_bridge.py running and the laptop on this hotspot?';
      err.hidden = false;
    } finally {
      $('connect-btn').disabled = false;
      $('connect-btn').textContent = 'Connect';
    }
  }

  /* ------------------------------------------------------------ dataset */
  async function loadDataset() {
    let data = null;
    const cached = store.get('grama.dataset', null);
    if (cached) {
      try { data = JSON.parse(cached); log(`Loaded ${data.X.length} listings from on-device storage`); } catch { data = null; }
    }
    if (!data) {
      log('Provisioning local shop data onto this device…');
      data = await api('/phone/dataset', { timeout: 20000 });
      delete data.__bytes;
      store.set('grama.dataset', JSON.stringify(data));
      log(`Stored ${data.X.length} listings on-device`);
    }
    S.data = data;
    renderDataset();
    emit('fl:data', data);
  }

  function renderDataset() {
    const d = S.data;
    const G = window.GramaCatalog;   // village labels / rupee prices, defined in shop.js
    $('store-name').textContent = d.store;
    $('data-count').textContent = d.X.length.toLocaleString();
    $('data-size').textContent = fmtBytes(JSON.stringify(d.raw).length);
    $('param-count').textContent = S.net.numParams.toLocaleString();

    const counts = {};
    d.raw.forEach(r => { counts[r.category] = (counts[r.category] || 0) + 1; });
    const bars = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    $('cat-bars').innerHTML = bars.map(([c, n]) => {
      const pct = Math.round(100 * n / d.raw.length);
      return `<div class="cat-bar"><span>${G.label(c)}</span><div class="track"><div style="width:${pct}%"></div></div><span>${pct}%</span></div>`;
    }).join('');

    $('data-table').querySelector('tbody').innerHTML = d.raw.slice(0, 6).map(r => `
      <tr><td>${G.id(r.listing_id)}</td><td>${G.label(r.category).split(' ')[0]}</td><td>${G.price(r.category, Number(r.price))}</td>
      <td>${r.views_7d}</td><td>${r.units_sold_next_7d}</td></tr>`).join('');
  }

  /* ------------------------------------------------------------ device */
  async function collectDevice() {
    const dev = {
      name: store.get('fl.devname', '') || 'Android phone',
      cores: navigator.hardwareConcurrency || null,
      memory_gb: navigator.deviceMemory || null,
      user_agent: navigator.userAgent,
    };
    try {
      if (navigator.userAgentData && navigator.userAgentData.getHighEntropyValues) {
        const h = await navigator.userAgentData.getHighEntropyValues(['model', 'platformVersion']);
        if (h.model) dev.model = h.model;
        if (h.platformVersion) dev.os = `${navigator.userAgentData.platform} ${h.platformVersion}`;
      }
    } catch { /* not supported */ }
    if (!dev.model) {
      const m = navigator.userAgent.match(/Android [\d.]+; ([^;)]+)/);
      dev.model = m ? m[1].trim() : (navigator.platform || 'unknown');
    }
    if (!store.get('fl.devname', '') && dev.model && dev.model !== 'K') dev.name = dev.model;
    S.device = dev;

    $('dev-model').textContent = dev.model || '–';
    $('dev-cores').textContent = dev.cores ?? '–';
    $('dev-mem').textContent = dev.memory_gb ? `${dev.memory_gb} GB` : '–';
    const conn = navigator.connection;
    $('dev-net').textContent = conn ? (conn.effectiveType || conn.type || '–') : 'Wi-Fi/hotspot';

    try {
      const batt = await navigator.getBattery();
      const upd = () => {
        S.device.battery = Math.round(batt.level * 100);
        S.device.charging = batt.charging;
        $('dev-batt').textContent = `${S.device.battery}%${batt.charging ? ' ⚡' : ''}`;
      };
      upd();
      batt.addEventListener('levelchange', upd);
      batt.addEventListener('chargingchange', upd);
    } catch { $('dev-batt').textContent = 'n/a'; }
    $('dev-wake').textContent = 'wakeLock' in navigator ? 'available' : 'n/a';
  }

  async function holdWakeLock(on) {
    try {
      if (on && 'wakeLock' in navigator && !S.wakeLock) {
        S.wakeLock = await navigator.wakeLock.request('screen');
        $('dev-wake').textContent = 'kept awake';
        S.wakeLock.addEventListener('release', () => { S.wakeLock = null; $('dev-wake').textContent = 'available'; });
      } else if (!on && S.wakeLock) {
        await S.wakeLock.release();
        S.wakeLock = null;
      }
    } catch { /* needs secure context; ignore */ }
  }

  /* ------------------------------------------------------------ polling */
  function startPolling() {
    stopPolling();
    const tick = async () => {
      try {
        const have = S.busy ? S.trainingRound : S.lastRound;
        const st = await api(`/phone/state?round=${have}`, { timeout: 15000 });
        S.failures = 0;
        S.state = st;
        setPill(st.running ? 'ok' : 'warn', st.running ? 'connected' : 'online');
        $('last-sync').textContent = new Date().toLocaleTimeString([], { hour12: false });
        renderState(st);
        if (st.weights && !S.busy) {
          S.bytesDown += st.__bytes;
          runRound(st);   // not awaited: keep polling while training
        }
        if (st.status === 'complete' && !S.finalShown) showFinal();
      } catch (e) {
        S.failures += 1;
        if (S.failures === 3) log(`Lost contact with laptop (${e.message})`, 'error');
        if (S.failures >= 3) setPill('err', 'offline');
      }
      S.pollTimer = setTimeout(tick, S.busy ? 2000 : 1000);
    };
    tick();
  }
  function stopPolling() { if (S.pollTimer) clearTimeout(S.pollTimer); S.pollTimer = null; }

  function setPill(cls, text) {
    const p = $('conn-pill');
    p.className = `pill ${cls}`;
    p.textContent = text;
  }

  const STATUS_TEXT = {
    idle: 'Not joined - tap “Join federation”',
    connecting: 'Connecting to federated server…',
    waiting_for_round: 'Joined. Waiting for the other clients and the next round…',
    weights_ready: 'New global model available',
    phone_training: 'Training on-device…',
    uploading: 'Sending update to server…',
    aggregating: 'Server is aggregating updates from all clients (FedAvg)…',
    complete: 'Federated training complete ✓',
    error: 'Connection to FL server failed - check the laptop',
  };
  const LOCAL_TEXT = {
    download: 'Downloading global model…',
    train: 'Training on-device with local shop data…',
    upload: 'Uploading model update (weights only)…',
  };

  function renderState(st) {
    $('round-num').textContent = st.round || 0;
    if (st.rounds_total) $('round-total').textContent = st.rounds_total;
    if (st.epochs) $('epoch-total').textContent = st.epochs;
    $('fl-id').textContent = st.fl_client_id ?? '–';
    $('fl-server').textContent = st.server || '–';

    const busyStatus = ['connecting', 'waiting_for_round', 'weights_ready', 'phone_training', 'uploading', 'aggregating'];
    const text = S.localPhase ? LOCAL_TEXT[S.localPhase] : (STATUS_TEXT[st.status] || st.status);
    const spinning = S.localPhase || busyStatus.includes(st.status);
    $('status-text').innerHTML = `${spinning ? '<span class="spin"></span>' : ''}${text}`;

    const jb = $('join-btn');
    jb.textContent = st.running ? 'Leave' : (st.status === 'complete' ? 'Join again' : 'Join federation');
    jb.className = `btn ${st.running ? 'btn-danger' : 'btn-primary'}`;

    let step = null;
    if (S.localPhase) step = S.localPhase;
    else if (['connecting', 'waiting_for_round'].includes(st.status)) step = 'wait';
    else if (['weights_ready'].includes(st.status)) step = 'download';
    else if (['uploading', 'aggregating'].includes(st.status)) step = 'aggregate';
    else if (st.status === 'complete') step = 'all';
    setStep(step);

    renderGlobal(st.global);
    emit('fl:state', { state: st, localPhase: S.localPhase });
  }

  function setStep(step) {
    const order = ['wait', 'download', 'train', 'upload', 'aggregate'];
    const idx = step === 'all' ? order.length : order.indexOf(step);
    document.querySelectorAll('#stepper li').forEach((li, i) => {
      li.classList.toggle('done', idx >= 0 && i < idx);
      li.classList.toggle('active', i === idx);
    });
  }

  /* ------------------------------------------------------------ training */
  async function runRound(st) {
    S.busy = true;
    const round = st.round;
    S.trainingRound = round;
    const epochs = st.epochs || 1;
    const pace = Number($('pace').value);
    const sim = $('sim-toggle').checked;
    const t0 = performance.now();
    $('mode-tag').textContent = sim ? 'simulated' : 'on-device';
    $('mode-tag').className = `tag ${sim ? 'sim' : ''}`;
    try {
      S.localPhase = 'download';
      renderState(S.state);
      log(`Round ${round}: downloaded global model (${fmtBytes(st.__bytes)})`);
      $('st-down').textContent = fmtBytes(S.bytesDown);
      S.net.loadState(st.weights);
      S.net.resetOptimizer();
      await sleep(500);

      S.localPhase = 'train';
      renderState(S.state);
      holdWakeLock(true);
      S.roundMarks.push({ at: S.batchCounter, round });
      const { X, y, y_std } = S.data;
      let lastPost = 0;
      let seen = 0;
      let simLoss = S.lossPts.length ? S.lossPts[S.lossPts.length - 1].loss : 1.0;

      for (let e = 1; e <= epochs; e++) {
        log(`Round ${round}: epoch ${e}/${epochs} - training on ${X.length} local listings`);
        const onBatch = b => {
          seen += b.samples;
          S.batchCounter += 1;
          S.lossPts.push({ x: S.batchCounter, loss: b.loss });
          const secs = (performance.now() - t0) / 1000;
          updateTrainUI(e, epochs, b, seen / secs, secs);
          const now = performance.now();
          if (now - lastPost > 500) {
            lastPost = now;
            api('/phone/progress', {
              method: 'POST',
              body: { round, epoch: e, epochs, batch: b.batch, batches: b.batches, loss: b.avgLoss },
            }).catch(() => {});
          }
        };

        let avg;
        if (sim) {
          const batches = Math.ceil(X.length / BATCH);
          let total = 0;
          for (let b = 1; b <= batches; b++) {
            simLoss = Math.max(0.05, simLoss * 0.992 + (Math.random() - 0.5) * 0.04);
            total += simLoss;
            onBatch({ batch: b, batches, loss: simLoss, avgLoss: total / b, samples: Math.min(BATCH, X.length - (b - 1) * BATCH) });
            await sleep(Math.max(pace, 10));
          }
          avg = total / batches;
        } else {
          avg = await S.net.trainEpoch(X, y, { lr: LR, batchSize: BATCH, paceMs: pace, onBatch });
        }

        const preds = S.net.predict(X);
        let mae = 0;
        for (let i = 0; i < y.length; i++) mae += Math.abs(preds[i] - y[i]);
        mae = (mae / y.length) * y_std;
        const secs = (performance.now() - t0) / 1000;
        $('st-loss').textContent = avg.toFixed(4);
        $('st-mae').textContent = `${mae.toFixed(2)} units`;
        log(`Round ${round}: epoch ${e} done - loss ${avg.toFixed(4)}, MAE ${mae.toFixed(2)} units`);
        await api('/phone/progress', {
          method: 'POST',
          body: { round, epoch: e, epochs, batch: 1, batches: 1, loss: avg, mae_units: mae,
                  samples_per_sec: seen / secs, epoch_done: true },
        }).catch(() => {});
      }

      S.localPhase = 'upload';
      renderState(S.state);
      const weights = sim ? st.weights : S.net.exportState();
      const body = JSON.stringify({
        round, weights, mode: sim ? 'simulated' : 'real', num_samples: X.length,
        duration_ms: Math.round(performance.now() - t0), final_loss: S.lossPts.length ? S.lossPts[S.lossPts.length - 1].loss : null,
        device: S.device,
      });
      await api('/phone/update', { method: 'POST', body, timeout: 30000 });
      S.bytesUp += body.length;
      $('st-up').textContent = fmtBytes(S.bytesUp);
      log(`Round ${round}: uploaded model update (${fmtBytes(body.length)}) - raw data stayed on device`);
      S.lastRound = round;
    } catch (e) {
      log(`Round ${round} failed: ${e.message}`, 'error');
      if (e.status === 409) S.lastRound = round;
    } finally {
      S.localPhase = null;
      S.busy = false;
      holdWakeLock(false);
      if (S.state) renderState(S.state);
    }
  }

  function updateTrainUI(epoch, epochs, b, sps, secs) {
    const ep = Math.round(100 * ((epoch - 1) + b.batch / b.batches) / epochs);
    const bp = Math.round(100 * b.batch / b.batches);
    $('epoch-num').textContent = epoch;
    $('epoch-total').textContent = epochs;
    $('epoch-pct').textContent = `${ep}%`;
    $('epoch-bar').style.width = `${ep}%`;
    $('batch-num').textContent = b.batch;
    $('batch-total').textContent = b.batches;
    $('batch-pct').textContent = `${bp}%`;
    $('batch-bar').style.width = `${bp}%`;
    $('st-loss').textContent = b.avgLoss.toFixed(4);
    $('st-sps').textContent = Math.round(sps).toLocaleString();
    $('st-time').textContent = `${secs.toFixed(1)} s`;
    if (S.batchCounter % 2 === 0) drawLoss();
  }

  /* ------------------------------------------------------------ charts */
  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    if (!cv.dataset.h) cv.dataset.h = cv.getAttribute('height');   // logical CSS height
    const w = cv.clientWidth, h = Number(cv.dataset.h);
    cv.style.height = `${h}px`;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function drawLoss() {
    const cv = $('loss-chart');
    const { ctx, w, h } = setupCanvas(cv);
    const pts = S.lossPts;
    const pad = { l: 34, r: 8, t: 10, b: 18 };
    ctx.font = '10px system-ui';
    ctx.fillStyle = '#6b7280';
    if (pts.length < 2) {
      ctx.fillText('Loss curve appears here during training', pad.l, h / 2);
      return;
    }
    const sorted = pts.map(p => p.loss).sort((a, b) => a - b);
    const ymax = Math.max(0.1, sorted[Math.floor(sorted.length * 0.98)] * 1.1);
    const x0 = pts[0].x, x1 = pts[pts.length - 1].x;
    const X = x => pad.l + (x - x0) / Math.max(1, x1 - x0) * (w - pad.l - pad.r);
    const Y = y => pad.t + (1 - Math.min(y, ymax) / ymax) * (h - pad.t - pad.b);

    ctx.strokeStyle = '#e4e7ec';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const yv = ymax * i / 3;
      ctx.beginPath(); ctx.moveTo(pad.l, Y(yv)); ctx.lineTo(w - pad.r, Y(yv)); ctx.stroke();
      ctx.fillText(yv.toFixed(2), 2, Y(yv) + 3);
    }
    ctx.setLineDash([3, 3]);
    S.roundMarks.forEach(m => {
      if (m.at < x0 - 1) return;
      ctx.beginPath(); ctx.moveTo(X(m.at), pad.t); ctx.lineTo(X(m.at), h - pad.b); ctx.stroke();
      ctx.fillText(`R${m.round}`, X(m.at) + 3, h - 5);
    });
    ctx.setLineDash([]);

    ctx.strokeStyle = 'rgba(37,99,235,.25)';
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.loss)) : ctx.moveTo(X(p.x), Y(p.loss))));
    ctx.stroke();

    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let ema = pts[0].loss;
    pts.forEach((p, i) => {
      ema = i ? 0.9 * ema + 0.1 * p.loss : ema;
      i ? ctx.lineTo(X(p.x), Y(ema)) : ctx.moveTo(X(p.x), Y(ema));
    });
    ctx.stroke();
  }

  function renderGlobal(g) {
    const hist = (g && g.history) || [];
    const last = hist[hist.length - 1];
    if (last) {
      $('g-mae').textContent = `${last.mae.toFixed(2)} u`;
      $('g-r2').textContent = last.r2.toFixed(3);
      $('g-clients').textContent = last.clients || '–';
    }
    const cv = $('global-chart');
    const { ctx, w, h } = setupCanvas(cv);
    ctx.font = '10px system-ui';
    ctx.fillStyle = '#6b7280';
    if (!hist.length) { ctx.fillText('Global test error per round appears here', 8, h / 2); return; }
    const total = Math.max((g.rounds_total || 0) + 1, hist.length);
    const maxMae = Math.max(...hist.map(r => r.mae)) * 1.1;
    const pad = { l: 8, r: 8, t: 14, b: 16 };
    const bw = (w - pad.l - pad.r) / total;
    hist.forEach((r, i) => {
      const bh = (r.mae / maxMae) * (h - pad.t - pad.b);
      const x = pad.l + i * bw + bw * 0.15;
      ctx.fillStyle = i === 0 ? '#cbd5e1' : (i === hist.length - 1 ? '#059669' : '#7c3aed');
      ctx.fillRect(x, h - pad.b - bh, bw * 0.7, bh);
      ctx.fillStyle = '#111827';
      ctx.fillText(r.mae.toFixed(1), x, h - pad.b - bh - 3);
      ctx.fillStyle = '#6b7280';
      ctx.fillText(i === 0 ? 'init' : `R${r.round}`, x, h - 3);
    });
  }

  /* ------------------------------------------------------------ final */
  async function fetchFinalModel(silent = false) {
    try {
      const fm = await api('/phone/final-model', { timeout: 15000 });
      const net = new FLTrainer.DemandNet();
      net.loadState(fm.weights);
      S.finalNet = net;
      emit('fl:model', net);
      return net;
    } catch (e) {
      if (!silent) log(`Could not fetch final model: ${e.message}`, 'error');
      return null;
    }
  }

  async function showFinal() {
    S.finalShown = true;
    const net = await fetchFinalModel();
    if (!net) return;
    const { X, raw, y_mean, y_std } = S.data;
    const pick = [];
    for (let i = 0; i < raw.length && pick.length < 6; i += Math.floor(raw.length / 6)) pick.push(i);
    const preds = net.predict(pick.map(i => X[i]));
    $('pred-table').querySelector('tbody').innerHTML = pick.map((i, k) => {
      const r = raw[i];
      const p = Math.max(0, preds[k] * y_std + y_mean);
      return `<tr><td>${window.GramaCatalog.id(r.listing_id)}</td><td>${window.GramaCatalog.price(r.category, Number(r.price))}</td><td>${r.units_sold_next_7d}</td><td>${p.toFixed(1)}</td></tr>`;
    }).join('');
    $('done-panel').hidden = false;
    log('Downloaded final global model - shop now shows on-device demand forecasts');
  }

  /* ------------------------------------------------------------ wiring */
  function wire() {
    $('join-btn').addEventListener('click', async () => {
      const running = S.state && S.state.running;
      try {
        if (running) {
          await api('/phone/leave', { method: 'POST' });
          log('Left the federation');
        } else {
          S.finalShown = false;
          S.lastRound = 0;
          $('done-panel').hidden = true;
          await api('/phone/join', { method: 'POST', body: { device: S.device } });
          log('Requested to join the federation');
        }
      } catch (e) { log(`Join/leave failed: ${e.message}`, 'error'); }
    });
    $('disconnect-btn').addEventListener('click', () => window.FLApp.showConnect());
    $('pace').value = store.get('fl.pace', '40');
    $('pace').addEventListener('change', () => store.set('fl.pace', $('pace').value));
    window.addEventListener('resize', () => { drawLoss(); if (S.state) renderGlobal(S.state.global); });

    window.addEventListener('beforeinstallprompt', e => {
      e.preventDefault();
      S.installEvt = e;
      $('install-btn').hidden = false;
    });
    $('install-btn').addEventListener('click', async () => {
      if (!S.installEvt) return;
      S.installEvt.prompt();
      await S.installEvt.userChoice;
      S.installEvt = null;
      $('install-btn').hidden = true;
    });

    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').catch(() => {
        log('Offline install unavailable on plain HTTP - use Chrome menu → "Add to Home screen" (see README)');
      });
    }
  }

  window.FLApp = {
    get data() { return S.data; },
    get finalNet() { return S.finalNet; },
    get state() { return S.state; },
    get localPhase() { return S.localPhase; },
    get device() { return S.device; },
    redraw() { drawLoss(); if (S.state) renderGlobal(S.state.global); },
    showConnect() { stopPolling(); showView('connect'); },
  };

  wire();
  initConnectForm();
  drawLoss();
  showView('connect');
  const savedHost = store.get('fl.host', '');
  if (savedHost) connect(savedHost, store.get('fl.port', '7000'));
})();
