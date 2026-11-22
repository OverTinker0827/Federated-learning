/* Grama Connect - mock village marketplace (browse, item detail, cart, account).
 *
 * The catalogue is this phone's own on-device listings (phone_data.json),
 * the same data the federated model trains on. The model's five underlying
 * categories are shown as village product groups (see CATALOG below); names,
 * pictures and rupee prices are derived deterministically from each listing.
 * Once a global model exists, item pages show an on-device demand forecast.
 * The federated-learning screen lives at #/ai (see app.js).
 */
(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };
  const money = v => `₹${Math.round(v).toLocaleString('en-IN')}`;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  /* ------------------------------------------------------------ catalogue */
  // key = category in the training data; scale turns its price into a plausible rupee price
  const CATALOG = {
    electronics: {
      label: 'Grains & Pulses', icon: '🌾', tint: '#fbf3dc', scale: 2.2,
      adj: ['Organic', 'Hand-pounded', 'Farm-fresh', 'Unpolished', 'Native', 'Stone-ground'],
      items: [['Basmati Rice', '🍚', '5 kg'], ['Sona Masoori Rice', '🍚', '5 kg'], ['Whole Wheat Atta', '🌾', '5 kg'],
              ['Ragi (Finger Millet)', '🌾', '1 kg'], ['Jowar', '🌾', '2 kg'], ['Bajra', '🌾', '2 kg'],
              ['Toor Dal', '🫘', '1 kg'], ['Moong Dal', '🫘', '1 kg'], ['Chana Dal', '🫘', '1 kg'],
              ['Rajma', '🫘', '1 kg'], ['Foxtail Millet', '🌾', '1 kg'], ['Red Rice', '🍚', '2 kg']],
    },
    fashion: {
      label: 'Handloom', icon: '🧵', tint: '#fde8e4', scale: 36,
      adj: ['Handwoven', 'Khadi', 'Block-printed', 'Natural-dye', 'Traditional', 'Handspun'],
      items: [['Cotton Kurta', '👕', '1 piece'], ['Cotton Saree', '🥻', '1 piece'], ['Dupatta', '🧣', '1 piece'],
              ['Jute Tote Bag', '👜', '1 piece'], ['Kolhapuri Chappals', '🩴', '1 pair'], ['Woollen Shawl', '🧣', '1 piece'],
              ['Ikat Bedsheet', '🛏️', 'double'], ['Phulkari Stole', '🧣', '1 piece'], ['Gamcha Towel Set', '🧵', 'set of 3'],
              ['Cotton Durrie Rug', '🧶', '4 × 6 ft']],
    },
    home: {
      label: 'Pottery & Wicker', icon: '🏺', tint: '#f9e6d8', scale: 14,
      adj: ['Handmade', 'Terracotta', 'Village-crafted', 'Hand-painted', 'Traditional', 'Eco-friendly'],
      items: [['Clay Water Pot (Matka)', '🏺', '10 L'], ['Diya Set', '🪔', 'set of 12'], ['Clay Cooking Pot', '🍲', '2 L'],
              ['Bamboo Basket', '🧺', 'medium'], ['Wicker Laundry Basket', '🧺', 'large'], ['Cane Chair', '🪑', '1 piece'],
              ['Terracotta Planter', '🪴', '8 inch'], ['Bamboo Floor Mat', '🎋', '3 × 6 ft'], ['Kulhad Tea Cups', '🍵', 'set of 6'],
              ['Coconut Shell Bowls', '🥥', 'set of 4'], ['Brass Oil Lamp', '🪔', '1 piece'], ['Palm-leaf Hand Fan', '🪭', 'set of 2']],
    },
    beauty: {
      label: 'Honey & Naturals', icon: '🍯', tint: '#fff1cc', scale: 28,
      adj: ['Pure', 'Natural', 'Cold-pressed', 'Farm-made', 'Herbal', 'Small-batch'],
      items: [['Forest Honey', '🍯', '500 g'], ['Desi Cow Ghee', '🧈', '500 ml'], ['Coconut Oil', '🥥', '1 L'],
              ['Neem Soap', '🧼', 'pack of 4'], ['Jaggery (Gur)', '🟫', '1 kg'], ['Mustard Oil', '🫙', '1 L'],
              ['Amla Hair Oil', '🌿', '200 ml'], ['Beeswax Candles', '🕯️', 'set of 3']],
    },
    sports: {
      label: 'Fruits & Spices', icon: '🥭', tint: '#e5f5e0', scale: 5,
      adj: ['Farm-fresh', 'Organic', 'Sun-dried', 'Hand-picked', 'Seasonal', 'Village-grown'],
      items: [['Alphonso Mangoes', '🥭', '1 dozen'], ['Bananas', '🍌', '1 dozen'], ['Guavas', '🍐', '1 kg'],
              ['Tender Coconuts', '🥥', 'pack of 4'], ['Jackfruit', '🍈', '1 whole'], ['Pomegranates', '🍎', '1 kg'],
              ['Red Chillies', '🌶️', '250 g'], ['Turmeric Root', '🫚', '500 g'], ['Black Pepper', '🧂', '250 g'],
              ['Green Cardamom', '🌿', '100 g'], ['Lemons', '🍋', '1 kg'], ['Tamarind', '🫘', '500 g']],
    },
  };
  const ORIGINS = ['Kutch, Gujarat', 'Wayanad, Kerala', 'Bastar, Chhattisgarh', 'Khurja, Uttar Pradesh', 'Madhubani, Bihar',
                   'Araku Valley, Andhra Pradesh', 'Coorg, Karnataka', 'Nashik, Maharashtra', 'Pochampally, Telangana',
                   'Majuli, Assam', 'Pattamadai, Tamil Nadu', 'Bishnupur, West Bengal'];
  const SELLERS = ['Self-help group', "Farmers' co-operative", 'Women artisans collective', 'Family farm', 'Tribal craft co-op'];

  function hash(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }

  let PRODUCTS = [];
  function buildProducts(data) {
    PRODUCTS = data.raw.map((r, i) => {
      const h = hash(r.listing_id);
      const cat = CATALOG[r.category] || CATALOG.home;
      const [base, emoji, pack] = cat.items[h % cat.items.length];
      const disc = Number(r.discount_pct) || 0;
      const mrp = Math.max(29, Number(r.price) * cat.scale);
      return {
        i, id: r.listing_id.replace(/^PR-/, 'GC-'), cat: r.category, emoji, tint: cat.tint, pack,
        name: `${cat.adj[(h >>> 8) % cat.adj.length]} ${base}`,
        price: mrp * (1 - disc / 100), oldPrice: disc ? mrp : null, disc,
        rating: Number(r.seller_rating), reviews: Number(r.num_reviews),
        ship: Number(r.shipping_days), free: Number(r.free_shipping) === 1, views: Number(r.views_7d),
        wish: Number(r.wishlist_7d),
        origin: ORIGINS[(h >>> 12) % ORIGINS.length], seller: SELLERS[(h >>> 16) % SELLERS.length],
        stock: 1 + ((h >>> 20) % 9),
      };
    });
  }

  // used by app.js so the On-device AI screen shows the same labels and prices
  window.GramaCatalog = {
    label: c => (CATALOG[c] ? CATALOG[c].label : c),
    price: (c, usd) => money(Math.max(29, usd * (CATALOG[c] ? CATALOG[c].scale : 1))),
    id: id => String(id).replace(/^PR-/, 'GC-'),
  };

  /* ---------------------------------------------- AI forecast (on-device) */
  const forecastCache = new Map();
  function forecast(p) {
    const net = window.FLApp && window.FLApp.finalNet;
    const data = window.FLApp && window.FLApp.data;
    if (!net || !data) return null;
    if (!forecastCache.has(p.i)) {
      const y = net.predict([data.X[p.i]])[0] * data.y_std + data.y_mean;
      forecastCache.set(p.i, Math.max(0, y));
    }
    return forecastCache.get(p.i);
  }
  const demandLabel = u => (u >= 20 ? 'High demand' : u >= 8 ? 'Steady demand' : 'Low demand');

  /* ------------------------------------------------------------ state */
  const UI = { cat: 'all', q: '', sort: 'popular', shown: 20, shopScroll: 0, lastOrder: null };
  let cart = store.get('grama.cart', {});
  let favs = new Set(store.get('grama.favs', []));
  let orders = store.get('grama.orders', []);

  const cartCount = () => Object.values(cart).reduce((a, b) => a + b, 0);
  function saveCart() { store.set('grama.cart', cart); renderBadges(); }
  function addToCart(i, qty = 1) {
    cart[i] = (cart[i] || 0) + qty;
    saveCart();
    toast(`Added to cart · ${cartCount()} item${cartCount() === 1 ? '' : 's'}`);
  }
  function toggleFav(i) {
    favs.has(i) ? favs.delete(i) : favs.add(i);
    store.set('grama.favs', [...favs]);
  }

  let toastTimer = null;
  function toast(msg) {
    const t = $('toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 1600);
  }

  function renderBadges() {
    const n = cartCount();
    document.querySelectorAll('[data-cart-badge]').forEach(b => { b.hidden = n === 0; b.textContent = n; });
  }

  const stars = r => '★'.repeat(Math.round(r)) + '☆'.repeat(5 - Math.round(r));

  /* ------------------------------------------------------------ shop home */
  function filtered() {
    const q = UI.q.trim().toLowerCase();
    let list = PRODUCTS.filter(p =>
      (UI.cat === 'all' || (UI.cat === 'saved' ? favs.has(p.i) : p.cat === UI.cat)) &&
      (!q || p.name.toLowerCase().includes(q) || CATALOG[p.cat].label.toLowerCase().includes(q) || p.origin.toLowerCase().includes(q)));
    const by = {
      popular: (a, b) => (b.views + 5 * b.wish) - (a.views + 5 * a.wish),
      'price-asc': (a, b) => a.price - b.price,
      'price-desc': (a, b) => b.price - a.price,
      deals: (a, b) => b.disc - a.disc,
      rating: (a, b) => b.rating - a.rating || b.reviews - a.reviews,
    }[UI.sort];
    return list.sort(by);
  }

  function renderChips() {
    const cats = ['all', ...Object.keys(CATALOG), 'saved'];
    $('cat-chips').innerHTML = cats.map(c => `<button class="chip ${UI.cat === c ? 'active' : ''}" data-cat="${c}">${
      c === 'all' ? 'All' : c === 'saved' ? '❤️ Saved' : `${CATALOG[c].icon} ${CATALOG[c].label}`}</button>`).join('');
  }

  function card(p) {
    const f = forecast(p);
    const inCart = cart[p.i] > 0;
    return `<div class="product">
      <a href="#/item/${p.i}" class="thumb" style="background:${p.tint}">
        ${p.disc ? `<span class="off">-${p.disc}%</span>` : ''}
        ${favs.has(p.i) ? '<span class="fav">❤️</span>' : ''}
        ${p.emoji}
      </a>
      <div class="pinfo">
        <a href="#/item/${p.i}" class="pname">${esc(p.name)}</a>
        <div class="pmeta">${esc(p.pack)} · ${esc(p.origin.split(',')[0])}</div>
        <div class="pprice">${money(p.price)}${p.oldPrice ? `<s>${money(p.oldPrice)}</s>` : ''}</div>
        <div class="pmeta">★ ${p.rating.toFixed(1)} · ${p.free ? 'Free delivery' : `${p.ship}-day delivery`}</div>
        ${f !== null && f >= 20 ? '<span class="ai-pill">🔥 AI: in demand</span>' : ''}
        <button class="add-mini ${inCart ? 'in' : ''}" data-add="${p.i}">${inCart ? `In cart (${cart[p.i]})` : 'Add to cart'}</button>
      </div>
    </div>`;
  }

  function renderShop() {
    renderChips();
    const list = filtered();
    $('result-count').textContent = `${list.length.toLocaleString('en-IN')} products`;
    $('promo').hidden = UI.cat !== 'all' || !!UI.q;
    $('product-grid').innerHTML = list.length
      ? list.slice(0, UI.shown).map(card).join('')
      : `<div class="empty" style="grid-column:1/-1"><div class="big-emoji">🔎</div>Nothing found</div>`;
    $('load-more').hidden = list.length <= UI.shown;
  }

  /* ------------------------------------------------------------ item page */
  function renderItem(i) {
    const p = PRODUCTS[i];
    if (!p) { location.hash = '#/shop'; return; }
    const f = forecast(p);
    let qty = 1;
    const forecastHtml = f !== null
      ? `<div class="forecast">🤖 <b>On-device AI forecast:</b> expected demand of <b>~${Math.round(f)} orders</b> in the next 7 days
           · ${demandLabel(f)}.<br><span class="small muted">Computed on this phone with the shared village-market model.</span></div>`
      : `<div class="forecast">🤖 Demand forecasts appear here once the shared model is trained.
           <a href="#/ai">Help train it on-device ›</a></div>`;
    $('item-body').innerHTML = `
      <div class="hero-img" style="background:${p.tint}">${p.emoji}</div>
      <div class="muted small">${CATALOG[p.cat].icon} ${CATALOG[p.cat].label} · ${p.id}</div>
      <h1 class="item-name">${esc(p.name)}</h1>
      <div class="rating">${stars(p.rating)} <span class="muted small">${p.rating.toFixed(1)} · ${p.reviews.toLocaleString('en-IN')} reviews</span></div>
      <div class="item-price" style="margin-top:8px">${money(p.price)}${p.oldPrice ? `<s>${money(p.oldPrice)}</s><span class="save">-${p.disc}%</span>` : ''}
        <span class="muted small" style="font-weight:400"> / ${esc(p.pack)}</span></div>
      <div class="facts">
        <div class="fact"><span class="muted">Sourced from</span><b>${esc(p.origin)}</b></div>
        <div class="fact"><span class="muted">Delivery</span><b>${p.free ? 'Free · ' : ''}${p.ship} days</b></div>
        <div class="fact"><span class="muted">Viewed this week</span><b>${p.views.toLocaleString('en-IN')} people</b></div>
        <div class="fact"><span class="muted">Stock</span><b>${p.stock === 1 ? 'Last one!' : `${p.stock} left`}</b></div>
      </div>
      ${forecastHtml}
      <div class="buy-row">
        <div class="qty"><button data-q="-1" aria-label="Less">−</button><span id="qty-val">1</span><button data-q="1" aria-label="More">+</button></div>
        <button class="btn btn-primary" id="add-btn">Add to cart</button>
        <button class="btn btn-ghost" id="fav-btn" style="flex:none;width:auto;margin:0" aria-label="Save">${favs.has(i) ? '❤️' : '🤍'}</button>
      </div>
      <p class="muted small" style="margin-top:14px">Sold by a ${esc(p.seller.toLowerCase())} in ${esc(p.origin)} via Grama Connect.
        ${p.wish} shoppers saved this item this week.</p>`;

    $('item-body').querySelectorAll('[data-q]').forEach(b => b.addEventListener('click', () => {
      qty = Math.min(p.stock, Math.max(1, qty + Number(b.dataset.q)));
      $('qty-val').textContent = qty;
    }));
    $('add-btn').addEventListener('click', () => addToCart(i, qty));
    $('fav-btn').addEventListener('click', () => {
      toggleFav(i);
      $('fav-btn').textContent = favs.has(i) ? '❤️' : '🤍';
      toast(favs.has(i) ? 'Saved to wishlist' : 'Removed from wishlist');
    });
  }

  /* ------------------------------------------------------------ cart */
  function totals() {
    const lines = Object.entries(cart).filter(([, q]) => q > 0).map(([i, q]) => ({ p: PRODUCTS[i], q })).filter(l => l.p);
    const subtotal = lines.reduce((s, l) => s + l.p.price * l.q, 0);
    const shipping = lines.length && lines.some(l => !l.p.free) && subtotal < 999 ? 49 : 0;
    return { lines, subtotal, shipping, total: subtotal + shipping };
  }

  function renderCart() {
    const body = $('cart-body');
    if (UI.lastOrder) {
      const o = UI.lastOrder;
      body.innerHTML = `<div class="card success"><div class="big-emoji">🎉</div>
        <h2 style="margin:8px 0 4px">Order placed!</h2>
        <p class="muted">Order <b>${o.id}</b> · ${o.items} item${o.items === 1 ? '' : 's'} · ${money(o.total)}</p>
        <p class="muted small">Your order supports village farmers and artisans.</p>
        <p class="demo-note">Demo order - nothing is charged or shipped.</p>
        <a href="#/shop" class="btn btn-primary btn-block" style="margin-top:12px">Keep shopping</a></div>`;
      UI.lastOrder = null;
      return;
    }
    const t = totals();
    if (!t.lines.length) {
      body.innerHTML = `<div class="empty"><div class="big-emoji">🧺</div><p>Your basket is empty</p>
        <a href="#/shop" class="btn btn-primary">Start browsing</a></div>`;
      return;
    }
    body.innerHTML = `
      <div class="card">${t.lines.map(({ p, q }) => `
        <div class="cart-item">
          <a href="#/item/${p.i}" class="cart-thumb" style="background:${p.tint}">${p.emoji}</a>
          <div class="cart-info">
            <a href="#/item/${p.i}" class="pname">${esc(p.name)}</a>
            <div class="muted small">${esc(p.pack)} · ${p.free ? 'Free delivery' : `${p.ship}-day delivery`}</div>
            <div class="cart-row">
              <div class="qty"><button data-dec="${p.i}">−</button><span>${q}</span><button data-inc="${p.i}">+</button></div>
              <b>${money(p.price * q)}</b>
            </div>
            <button class="link-btn" data-rm="${p.i}">Remove</button>
          </div>
        </div>`).join('')}
      </div>
      <div class="card">
        <div class="sum-row"><span>Subtotal</span><span>${money(t.subtotal)}</span></div>
        <div class="sum-row"><span>Delivery</span><span>${t.shipping ? money(t.shipping) : 'Free'}</span></div>
        <div class="sum-row total"><span>Total</span><span>${money(t.total)}</span></div>
        <button class="btn btn-primary btn-block" id="checkout-btn" style="margin-top:12px">Place demo order</button>
        <p class="demo-note">Demo checkout - no payment details are collected. Free delivery above ₹999.</p>
      </div>`;

    body.querySelectorAll('[data-inc]').forEach(b => b.addEventListener('click', () => {
      const i = b.dataset.inc;
      cart[i] = Math.min(PRODUCTS[i].stock, cart[i] + 1); saveCart(); renderCart();
    }));
    body.querySelectorAll('[data-dec]').forEach(b => b.addEventListener('click', () => {
      const i = b.dataset.dec;
      cart[i] -= 1; if (cart[i] <= 0) delete cart[i]; saveCart(); renderCart();
    }));
    body.querySelectorAll('[data-rm]').forEach(b => b.addEventListener('click', () => {
      delete cart[b.dataset.rm]; saveCart(); renderCart();
    }));
    $('checkout-btn').addEventListener('click', () => {
      const o = { id: `GC-${Date.now().toString(36).toUpperCase().slice(-6)}`, items: cartCount(), total: t.total, at: Date.now() };
      orders = [o, ...orders].slice(0, 20);
      store.set('grama.orders', orders);
      cart = {};
      saveCart();
      UI.lastOrder = o;
      renderCart();
    });
  }

  /* ------------------------------------------------------------ account */
  function aiStatusText() {
    const st = window.FLApp && window.FLApp.state;
    if (window.FLApp && window.FLApp.localPhase === 'train') return `Training now · round ${st ? st.round : ''}`;
    if (!st) return 'Not connected';
    if (st.status === 'complete') return 'Model trained ✓';
    if (st.running) return `Joined · round ${st.round || 0}/${st.rounds_total || '–'}`;
    return window.FLApp.finalNet ? 'Model ready · tap to train again' : 'Not joined';
  }

  function renderAccount() {
    const dev = (window.FLApp && window.FLApp.device) || {};
    $('account-body').innerHTML = `
      <div class="card profile">
        <div class="avatar">🙏</div>
        <div><b>Namaste!</b><div class="muted small">${esc(dev.name || 'Android phone')}${dev.model && dev.model !== dev.name ? ` · ${esc(dev.model)}` : ''}</div></div>
      </div>

      <div class="section-label">Shopping</div>
      <div class="card menu">
        <a href="#/cart"><span class="mi-icon">🧺</span><span class="mi-text">Basket<span class="mi-sub">${cartCount()} item(s)</span></span><span class="chev">›</span></a>
        <button id="saved-link"><span class="mi-icon">❤️</span><span class="mi-text">Wishlist<span class="mi-sub">${favs.size} saved</span></span><span class="chev">›</span></button>
        <button id="orders-toggle"><span class="mi-icon">📦</span><span class="mi-text">Orders<span class="mi-sub">${orders.length} demo order(s)</span></span><span class="chev">›</span></button>
      </div>
      <div class="card" id="orders-list" hidden>${orders.length ? orders.map(o => `
        <div class="order"><span><b>${o.id}</b><br><span class="muted small">${new Date(o.at).toLocaleString('en-IN')}</span></span>
        <span>${o.items} item(s)<br><b>${money(o.total)}</b></span></div>`).join('') : '<div class="muted">No orders yet</div>'}</div>

      <div class="section-label">Seller hub</div>
      <div class="card menu">
        <a href="#/ai"><span class="mi-icon">🤖</span><span class="mi-text">On-device AI training<span class="mi-sub" id="ai-sub">${aiStatusText()}</span></span><span class="chev">›</span></a>
        <button id="conn-link"><span class="mi-icon">💻</span><span class="mi-text">Laptop connection<span class="mi-sub">Change the training server IP</span></span><span class="chev">›</span></button>
      </div>
      <p class="demo-note">Grama Connect is a demo village marketplace for the federated learning project.</p>`;

    $('orders-toggle').addEventListener('click', () => { $('orders-list').hidden = !$('orders-list').hidden; });
    $('saved-link').addEventListener('click', () => { UI.cat = 'saved'; UI.shown = 20; location.hash = '#/shop'; });
    $('conn-link').addEventListener('click', () => window.FLApp.showConnect());
  }

  /* ------------------------------------------------------------ training chip */
  function renderTrainChip() {
    const chip = $('train-chip');
    const st = window.FLApp && window.FLApp.state;
    const ai = $('ai-sub');
    if (ai) ai.textContent = aiStatusText();
    if (!st) { chip.hidden = true; return; }
    chip.classList.toggle('done', st.status === 'complete');
    if (window.FLApp.localPhase === 'train') {
      chip.hidden = false; $('train-chip-text').textContent = `AI training · R${st.round}/${st.rounds_total || '–'}`;
    } else if (st.running) {
      chip.hidden = false; $('train-chip-text').textContent = `AI · round ${st.round || 0}/${st.rounds_total || '–'}`;
    } else if (st.status === 'complete') {
      chip.hidden = false; $('train-chip-text').textContent = 'AI model ready';
    } else {
      chip.hidden = true;
    }
  }

  /* ------------------------------------------------------------ router */
  const PAGES = ['shop', 'item', 'cart', 'account', 'ai'];
  let current = null;

  function route() {
    if ($('view-app').hidden) return;
    const parts = (location.hash || '#/shop').replace(/^#\/?/, '').split('/');
    const page = PAGES.includes(parts[0]) ? parts[0] : 'shop';
    if (current === 'shop') UI.shopScroll = window.scrollY;

    PAGES.forEach(pg => { $(`page-${pg}`).hidden = pg !== page; });
    const tab = { shop: 'shop', item: 'shop', cart: 'cart', account: 'account', ai: 'account' }[page];
    document.querySelectorAll('#tabbar a').forEach(a => a.classList.toggle('active', a.dataset.tab === tab));

    if (page === 'shop') renderShop();
    if (page === 'item') renderItem(Number(parts[1]));
    if (page === 'cart') renderCart();
    if (page === 'account') renderAccount();
    if (page === 'ai') requestAnimationFrame(() => window.FLApp.redraw());

    window.scrollTo(0, page === 'shop' ? UI.shopScroll : 0);
    current = page;
  }

  /* ------------------------------------------------------------ wiring */
  $('cat-chips').addEventListener('click', e => {
    const b = e.target.closest('[data-cat]');
    if (!b) return;
    UI.cat = b.dataset.cat; UI.shown = 20; renderShop();
  });
  $('product-grid').addEventListener('click', e => {
    const b = e.target.closest('[data-add]');
    if (!b) return;
    const i = Number(b.dataset.add);
    if (cart[i] >= PRODUCTS[i].stock) { toast('No more stock for this item'); return; }
    addToCart(i);
    b.classList.add('in');
    b.textContent = `In cart (${cart[i]})`;
  });
  let searchTimer = null;
  $('search').addEventListener('input', e => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { UI.q = e.target.value; UI.shown = 20; renderShop(); }, 150);
  });
  $('sort').addEventListener('change', e => { UI.sort = e.target.value; UI.shown = 20; renderShop(); });
  $('load-more').addEventListener('click', () => { UI.shown += 20; renderShop(); });
  $('item-back').addEventListener('click', e => { if (history.length > 1) { e.preventDefault(); history.back(); } });

  window.addEventListener('hashchange', route);
  document.addEventListener('fl:data', e => { buildProducts(e.detail); forecastCache.clear(); renderBadges(); });
  document.addEventListener('fl:shown', () => { current = null; route(); });
  document.addEventListener('fl:state', renderTrainChip);
  document.addEventListener('fl:model', () => {
    forecastCache.clear();
    if (current === 'shop') renderShop();
    if (current === 'item') route();
  });
})();
