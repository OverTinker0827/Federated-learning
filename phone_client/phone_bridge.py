#!/usr/bin/env python3
"""
Phone Bridge - lets an Android phone take part in federated learning.

A phone browser cannot open raw TCP sockets or read torch.save() payloads,
so this small Flask service runs on the laptop and sits between the two:

    Phone web app  <--HTTP/JSON-->  phone_bridge.py  <--TCP/torch-->  server/server.py

* To the FL server it is an ordinary client (same wire protocol as
  client*/client_com.py), so server.py needs no special cases.
* To the phone it exposes a tiny JSON API (/phone/*): download the global
  weights, report training progress, upload the locally trained weights.
* To the React dashboard it exposes the same API as the laptop clients
  (/status, /start, /stop, /logs, /logs/clear), so the phone shows up as a
  normal client card.
* It also serves the installable phone web app itself (phone_client/app/).

The actual training happens on the phone (phone_client/app/trainer.js).

Usage:
    python phone_client/phone_bridge.py                 # listens on 0.0.0.0:7000
    python phone_client/phone_bridge.py --port 7000 --server-ip 127.0.0.1
"""

import argparse
import io
import json
import os
import socket
import struct
import threading
import time
from datetime import datetime

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP_DIR = os.path.join(HERE, "app")
PHONE_DATA = os.path.join(HERE, "phone_data.json")
SERVER_CONFIG = os.path.join(ROOT, "server", "server_config.json")
ROUND_METRICS = os.path.join(ROOT, "server", "round_metrics.json")

DASHBOARD_CLIENT_ID = 3   # which card the phone occupies in dashboard/src/client_config.json


def state_dict_to_json(sd):
    return {k: {"shape": list(v.shape), "data": v.detach().cpu().float().flatten().tolist()} for k, v in sd.items()}


def json_to_state_dict(js, template):
    out = {}
    for k, ref in template.items():
        if k not in js:
            raise ValueError(f"missing parameter {k}")
        t = torch.tensor(js[k]["data"], dtype=ref.dtype).reshape(ref.shape)
        if not torch.isfinite(t).all():
            raise ValueError(f"non-finite values in {k}")
        out[k] = t
    return out


def local_ips():
    """Best-effort list of this laptop's LAN addresses (for the console banner)."""
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))   # no packet is sent; just picks the default interface
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


class PhoneBridge:
    def __init__(self, server_ip, fl_port):
        self.server_ip = server_ip
        self.fl_port = fl_port
        self.lock = threading.RLock()
        self.logs = []
        self.max_logs = 1000
        self.gen = 0
        self._reset()

    # ------------------------------------------------------------------ state
    def _reset(self):
        self.running = False            # bridge connected (or connecting) to FL server
        self.status = "idle"
        self.fl_client_id = None
        self.round = 0
        self.global_weights = None      # JSON form, sent to the phone
        self.template = None            # torch state_dict, to rebuild tensors
        self.update = None              # JSON weights uploaded by the phone
        self.update_event = threading.Event()
        self.stop_event = threading.Event()
        self.sock = None
        self.epochs = None
        self.progress = {}
        self.phone_last_seen = 0.0
        self.device = {}
        self.round_reports = []
        self.weights_sent_at = None

    def log(self, message, level="INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] [{level}] {message}"
        with self.lock:
            self.logs.append(entry)
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]
        print(entry, flush=True)

    def server_config(self):
        try:
            with open(SERVER_CONFIG) as fh:
                return json.load(fh)
        except Exception:
            return {}

    def global_metrics(self):
        try:
            with open(ROUND_METRICS) as fh:
                return json.load(fh)
        except Exception:
            return None

    def phone_online(self):
        return time.time() - self.phone_last_seen < 6

    # --------------------------------------------------------------- FL link
    def start(self, server_ip=None, fl_port=None, epochs=None, source="dashboard"):
        with self.lock:
            if self.running:
                return False, "already running"
            device, last_seen = self.device, self.phone_last_seen   # phone identity survives a restart
            self._reset()
            self.device, self.phone_last_seen = device, last_seen
            if server_ip:
                self.server_ip = server_ip
            if fl_port:
                self.fl_port = int(fl_port)
            self.epochs = int(epochs) if epochs else int(self.server_config().get("epochs", 2))
            self.running = True
            self.status = "connecting"
            self.gen += 1
            gen, stop_ev, upd_ev = self.gen, self.stop_event, self.update_event
        self.log(f"Phone client started from {source}; connecting to FL server {self.server_ip}:{self.fl_port}")
        threading.Thread(target=self._fl_loop, args=(gen, stop_ev, upd_ev), daemon=True).start()
        return True, "started"

    def stop(self, reason="stopped"):
        with self.lock:
            if not self.running:
                return False
            self.stop_event.set()
            self.update_event.set()
            sock = self.sock
        try:
            if sock:
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
        except Exception:
            pass
        with self.lock:
            self.running = False
            self.status = "idle"
        self.log(f"Phone client {reason}")
        return True

    @staticmethod
    def _recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("FL server closed the connection")
            buf += chunk
        return buf

    def _fl_loop(self, gen, stop_event, update_event):
        s = None
        try:
            s = socket.create_connection((self.server_ip, self.fl_port), timeout=10)
            s.settimeout(None)
            with self.lock:
                self.sock = s
            self.fl_client_id = struct.unpack("!I", self._recv_exact(s, 4))[0]
            with self.lock:
                self.status = "waiting_for_round"
            self.log(f"Connected to FL server - assigned federated client ID {self.fl_client_id}")

            while not stop_event.is_set():
                size = struct.unpack("!Q", self._recv_exact(s, 8))[0]
                if size == 0:
                    with self.lock:
                        self.status = "complete"
                    self.log("Server signalled training complete")
                    break
                sd = torch.load(io.BytesIO(self._recv_exact(s, size)), map_location="cpu")
                with self.lock:
                    self.round += 1
                    self.template = sd
                    self.global_weights = state_dict_to_json(sd)
                    self.update = None
                    update_event.clear()
                    self.status = "weights_ready"
                    self.progress = {}
                    self.weights_sent_at = time.time()
                self.log(f"Round {self.round}: received global model ({size / 1024:.1f} KB) - waiting for phone to download")
                if not self.phone_online():
                    self.log("Phone is not polling right now - open the phone app and tap 'Join federation'", "WARNING")

                # Wait until the phone uploads its locally trained weights
                while not update_event.wait(timeout=1.0):
                    if stop_event.is_set():
                        break
                if stop_event.is_set():
                    break

                with self.lock:
                    upd = self.update
                    self.status = "uploading"
                new_sd = json_to_state_dict(upd, self.template)
                buf = io.BytesIO()
                torch.save(new_sd, buf)
                payload = buf.getvalue()
                s.sendall(struct.pack("!Q", len(payload)))
                s.sendall(payload)
                with self.lock:
                    self.status = "aggregating"
                self.log(f"Round {self.round}: forwarded phone update to server ({len(payload) / 1024:.1f} KB) - waiting for aggregation")

        except Exception as e:
            if not stop_event.is_set():
                self.log(f"FL link error: {e}", "ERROR")
                with self.lock:
                    if self.gen == gen:
                        self.status = "error"
        finally:
            try:
                if s:
                    s.close()
            except Exception:
                pass
            with self.lock:
                if self.gen != gen:
                    return  # a newer session has started; leave its state alone
                self.running = False
                if self.status not in ("complete", "error"):
                    self.status = "idle"
            self.log(f"Phone client exited after {self.round} round(s)")

    # ------------------------------------------------------------ phone side
    def touch(self, device=None):
        with self.lock:
            first = not self.phone_online()
            self.phone_last_seen = time.time()
            if device:
                self.device = device
        if first:
            name = (device or self.device or {}).get("name", "phone")
            self.log(f"Phone app connected ({name}) from {request.remote_addr}")

    def receive_progress(self, p):
        with self.lock:
            prev_epoch = self.progress.get("epoch")
            self.progress = p
            if self.status == "weights_ready":
                self.status = "phone_training"
        if p.get("epoch_done"):
            self.log(f"Phone | Round {p.get('round')} | Epoch {p.get('epoch')}/{p.get('epochs')} | "
                     f"Loss: {p.get('loss', 0):.4f} | MAE: {p.get('mae_units', 0):.2f} units | "
                     f"{p.get('samples_per_sec', 0):.0f} samples/s")
        elif prev_epoch != p.get("epoch") and p.get("batch", 0) <= 1:
            self.log(f"Phone | Round {p.get('round')} | training epoch {p.get('epoch')}/{p.get('epochs')} on-device")

    def receive_update(self, body):
        with self.lock:
            if self.status not in ("weights_ready", "phone_training"):
                return False, f"not expecting an update (status={self.status})"
            if int(body.get("round", -1)) != self.round:
                return False, f"stale round {body.get('round')} (current {self.round})"
            # validate before accepting
            json_to_state_dict(body["weights"], self.template)
            self.update = body["weights"]
            self.round_reports.append({
                "round": self.round,
                "num_samples": body.get("num_samples"),
                "duration_ms": body.get("duration_ms"),
                "final_loss": body.get("final_loss"),
                "mode": body.get("mode", "real"),
            })
            self.update_event.set()
        self.log(f"Round {self.round}: phone uploaded weights - {body.get('num_samples')} samples, "
                 f"{(body.get('duration_ms') or 0) / 1000:.1f}s on-device, mode={body.get('mode', 'real')}")
        return True, "ok"

    def phone_state(self, have_round):
        with self.lock:
            cfg = self.server_config()
            st = {
                "status": self.status,
                "running": self.running,
                "round": self.round,
                "rounds_total": cfg.get("rounds"),
                "epochs": self.epochs or cfg.get("epochs", 2),
                "fl_client_id": self.fl_client_id,
                "server": f"{self.server_ip}:{self.fl_port}",
                "global": self.global_metrics(),
            }
            if self.global_weights is not None and self.round > have_round and self.status in ("weights_ready", "phone_training"):
                st["weights"] = self.global_weights
            return st

    def dashboard_status(self):
        with self.lock:
            p = self.progress or {}
            pct = 0
            if self.status == "phone_training" and p.get("epochs"):
                done = (p.get("epoch", 1) - 1) + (p.get("batch", 0) / max(1, p.get("batches", 1)))
                pct = int(100 * done / p["epochs"])
            return {
                "client_id": DASHBOARD_CLIENT_ID,
                "type": "phone",
                "running": self.running,
                "training": self.status == "phone_training",
                "progress": pct,
                "phase": self.status,
                "round": self.round,
                "phone_online": self.phone_online(),
                "device": self.device,
                "timestamp": datetime.now().isoformat(),
            }


def create_app(bridge):
    app = Flask(__name__, static_folder=None)
    CORS(app)

    # ---------------------------------------------------- phone web app files
    @app.route("/")
    def index():
        return send_from_directory(APP_DIR, "index.html")

    @app.route("/sw.js")
    def service_worker():
        resp = send_from_directory(APP_DIR, "sw.js")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Service-Worker-Allowed"] = "/"
        return resp

    @app.route("/app/<path:fname>")
    def app_files(fname):
        return send_from_directory(APP_DIR, fname)

    @app.route("/manifest.json")
    def manifest():
        return send_from_directory(APP_DIR, "manifest.json", mimetype="application/manifest+json")

    # ----------------------------------------------------------- phone API
    @app.route("/phone/hello")
    def hello():
        bridge.touch()
        cfg = bridge.server_config()
        return jsonify({
            "bridge": "fl-phone-bridge",
            "server": f"{bridge.server_ip}:{bridge.fl_port}",
            "rounds": cfg.get("rounds"),
            "epochs": cfg.get("epochs"),
            "num_clients": cfg.get("num_clients"),
            "model": {"name": "DemandNet", "layers": [16, 64, 32, 1]},
        })

    @app.route("/phone/dataset")
    def dataset():
        if not os.path.exists(PHONE_DATA):
            return jsonify({"error": "phone_data.json missing - run generate_ecommerce_data.py"}), 404
        return send_from_directory(HERE, "phone_data.json", mimetype="application/json")

    @app.route("/phone/final-model")
    def final_model():
        """Final aggregated global model, so the phone can run inference on-device."""
        path = os.path.join(ROOT, "server", "global_weights.pth")
        if not os.path.exists(path):
            return jsonify({"error": "no global model yet"}), 404
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            return jsonify({"weights": state_dict_to_json(sd), "saved_at": os.path.getmtime(path)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/phone/join", methods=["POST"])
    def join():
        body = request.get_json(silent=True) or {}
        bridge.touch(body.get("device"))
        if bridge.running:
            return jsonify({"message": "already joined", "status": bridge.status})
        bridge.start(source="phone")
        return jsonify({"message": "joined", "status": bridge.status})

    @app.route("/phone/leave", methods=["POST"])
    def leave():
        bridge.stop("left the federation (from phone)")
        return jsonify({"message": "left"})

    @app.route("/phone/state")
    def state():
        bridge.touch()
        return jsonify(bridge.phone_state(request.args.get("round", 0, type=int)))

    @app.route("/phone/progress", methods=["POST"])
    def progress():
        bridge.touch()
        bridge.receive_progress(request.get_json(silent=True) or {})
        return jsonify({"ok": True})

    @app.route("/phone/update", methods=["POST"])
    def update():
        bridge.touch()
        try:
            ok, msg = bridge.receive_update(request.get_json(force=True))
        except Exception as e:
            bridge.log(f"Rejected phone update: {e}", "ERROR")
            return jsonify({"error": str(e)}), 400
        return (jsonify({"message": msg}), 200) if ok else (jsonify({"error": msg}), 409)

    # ------------------------------------------- dashboard-compatible client API
    @app.route("/status")
    def status():
        return jsonify(bridge.dashboard_status())

    @app.route("/start", methods=["POST"])
    def start():
        body = request.get_json(silent=True) or {}
        ok, msg = bridge.start(body.get("server_ip"), body.get("fl_port"), body.get("epochs"), source="dashboard")
        if not ok:
            return jsonify({"error": "Phone client is already running"}), 400
        return jsonify({"message": "Phone client started - waiting for the phone to train",
                        "client_id": DASHBOARD_CLIENT_ID})

    @app.route("/stop", methods=["POST"])
    def stop():
        if not bridge.stop("stopped from dashboard"):
            return jsonify({"error": "Client is not running"}), 400
        return jsonify({"message": "Client stopped"})

    @app.route("/logs")
    def logs():
        offset = request.args.get("offset", 0, type=int)
        with bridge.lock:
            lines = bridge.logs[offset:]
            total = len(bridge.logs)
        return jsonify({"logs": "\n".join(lines), "total_lines": total, "returned_lines": len(lines)})

    @app.route("/logs/clear", methods=["POST"])
    def clear_logs():
        with bridge.lock:
            bridge.logs = []
        bridge.log("Logs cleared")
        return jsonify({"message": "Logs cleared"})

    return app


def main():
    ap = argparse.ArgumentParser(description="Bridge between the phone web app and the FL server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--server-ip", default="127.0.0.1", help="FL server address as seen from this laptop")
    ap.add_argument("--fl-port", type=int, default=8765)
    args = ap.parse_args()

    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # hide per-request polling noise

    bridge = PhoneBridge(args.server_ip, args.fl_port)
    app = create_app(bridge)

    print("\n" + "=" * 64)
    print("Federated Learning - Phone Bridge")
    print("=" * 64)
    print(f"FL server:   {args.server_ip}:{args.fl_port}")
    for ip in local_ips() or ["<laptop-ip>"]:
        print(f"Phone app:   http://{ip}:{args.port}")
    print("Open that URL in Chrome on the phone (laptop must be on the phone's hotspot).")
    print("=" * 64 + "\n")
    bridge.log("Phone bridge initialised")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
