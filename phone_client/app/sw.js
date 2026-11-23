/* Grama Connect service worker: caches the app shell so the installed app opens
   instantly (and offline) to the connect screen. API calls are never cached. */
const CACHE = 'grama-connect-v4';
const SHELL = ['/', '/manifest.json', '/app/styles.css', '/app/app.js', '/app/trainer.js', '/app/shop.js',
               '/app/icon-192.png', '/app/icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin || url.pathname.startsWith('/phone/')) return;
  // network-first so code changes on the laptop show up; fall back to cache offline
  e.respondWith(
    fetch(e.request)
      .then(res => { const copy = res.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); return res; })
      .catch(() => caches.match(e.request))
  );
});
