# Federated-learning

## Phone client demo: e-commerce demand forecasting

Two laptop "sellers" and an **Android phone** train one shared model together
with Federated Averaging. The phone really trains: it runs the same small
network as PyTorch, written in plain JavaScript in the browser, and only model
weights leave the device.

**Task:** predict how many units a marketplace listing will sell in the next 7 days.
**Model:** `DemandNet`, a 16 → 64 → 32 → 1 MLP with 3,201 parameters (`server/model.py`).

| Participant | Store (mock data) | Runs on |
|---|---|---|
| client1 | TechHub Electronics (1,600 listings) | laptop, PyTorch |
| client2 | Urban Threads Boutique (1,500 listings) | laptop, PyTorch |
| client3 (dashboard card) | Grama Connect, village marketplace (1,200 listings) | **phone**, JS in Chrome |
| server | 1,500-listing held-out test set | laptop |

```
Phone (Grama Connect web app) --HTTP/JSON--> phone_client/phone_bridge.py :7000 --TCP--> server/server.py :8765
client1 / client2 (laptop, PyTorch) ------------------------------------------------> server/server.py
Dashboard (React :3000) --> server API :5000, client APIs :6001/:6002, phone bridge :7000
```

A browser can't open raw TCP sockets or read `torch.save` data. So
`phone_bridge.py` acts as a normal FL client toward the server and converts the
weights to and from JSON for the phone. It also serves the phone app itself.

### 1. One-time setup
```bash
.venv/bin/python generate_ecommerce_data.py     # mock datasets + ecommerce_scaler.json (already committed)
```

### 2. Network
1. Turn on the **phone's hotspot** and connect the laptop to it.
2. If the laptop firewall is on, allow port 7000 (the only port the phone uses),
   e.g. `sudo ufw allow 7000/tcp`.

### 3. Start everything on the laptop
```bash
./start_demo.sh                 # server API, client1, client2, phone bridge
cd dashboard && npm start       # in a second terminal
```
The `[phone]` lines print the URL for the phone, e.g. `http://192.168.43.12:7000`.

### 4. On the phone (Chrome on Android)
1. Open `http://<laptop-ip>:7000`, enter the laptop IP, and tap **Connect**.
2. Install it on the home screen. Pick one:
   * **Quick:** Chrome menu ⋮ → **Add to Home screen**. This works over plain HTTP
     and opens like an app.
   * **Full install (standalone, works offline):** Chrome needs a "secure" origin.
     On the phone, open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`,
     add `http://<laptop-ip>:7000`, set it to *Enabled*, and relaunch Chrome. Reload
     the page; the app's **Install app** button / Chrome's **Install app** then appears.
3. When the installed app opens, it shows the IP screen (the last IP is remembered).

### 5. The phone app
The phone runs **Grama Connect**, a mock Indian village marketplace (light theme)
with five groups: Grains & Pulses, Handloom, Pottery & Wicker, Honey & Naturals,
and Fruits & Spices. Prices are in ₹.
* **Shop:** browse, search, filter by group, sort, and save favourites. The
  catalogue is the phone's own 1,200 listings, the same data it trains on. The
  model's five underlying categories are shown under village names, and each
  listing gets a product name, pack size, origin village and rupee price derived
  from its ID.
* **Item page:** details, quantity and add to cart. Once a trained global model
  exists, it shows an **on-device AI demand forecast**, and the grid marks
  "🔥 AI: in demand" items.
* **Basket:** change quantities, see totals, and place a **demo order** (no
  payment details, nothing is charged).
* **Account → Seller hub → On-device AI training:** the federated-learning
  screen. While training runs, a chip in the shop header shows the round and
  links here.

### 6. Run a round
In the dashboard, set `num_clients = 3` (the default), click **Start Server**, then
**Start All** on the Clients tab. Alternatively, tap **Join federation** on the
phone's On-device AI screen. The phone trains automatically each round, even
while you keep browsing the shop. The On-device AI screen shows:
* the round progress steps (download → train → upload → aggregate)
* live loss chart and samples/s
* how much was downloaded and uploaded
* the global model's test error per round
* at the end, predictions made on-device with the final global model

In a typical run, the global test MAE falls from about 14 units (untrained) to
about 4.4 units (R² ≈ 0.9) after 5 rounds.

Phone settings:
* **Demo pacing:** adds a small delay per batch so training is watchable.
* **Simulated training:** fallback that animates training and sends the
  weights back unchanged.

Keep the phone screen on during training. The app requests a wake lock where
Chrome allows it, but Android slows down background tabs.

### Files
* `generate_ecommerce_data.py`, `ecommerce_features.py`: mock data, shared feature encoding
* `phone_client/phone_bridge.py`: phone ↔ FL server bridge, plus the dashboard client API
* `phone_client/app/`: the installable web app (`shop.js` is the storefront, `app.js` the on-device AI screen, `trainer.js` the JS training code)
* `start_demo.sh`: starts the laptop-side processes
* **Model Builder** (dashboard tab): choose hidden layers, neurons per layer and activations (ReLU, Leaky ReLU, Tanh, Sigmoid, Linear), or load *Preset 1 - DemandNet*. Saved to `server/model_config.json` (stop the server first). The server sends the architecture with each round's weights, so laptop clients and the phone build the same network automatically.

---

## Original setup notes (cloud / VM deployment)


RUN dashboard using 
```
npm start
```

Configure dashboard ip to public ip in .env file

RUN each client using 
```bash
python3 .clientx/run_api.py
```
 in its directory.

RUN server api endpoints using
 ```bash
python3 ./server/run_api.py
```

 Change the client ips in dashboard/src/client_config.json with the private ips.
 
 Change the server ip in client_/config.py to its private ip

