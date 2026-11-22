#!/usr/bin/env bash
# Starts every laptop-side process for the phone federated-learning demo:
#   server API (:5000)  client1 API (:6001)  client2 API (:6002)  phone bridge (:7000)
# Then start the dashboard separately:  cd dashboard && npm start
# Ctrl+C stops everything.
set -u
cd "$(dirname "$0")"
ROOT="$PWD"
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python3

trap 'echo; echo "Stopping demo..."; kill 0' EXIT INT TERM

(cd "$ROOT/server"  && exec "$PY" -u run_api.py) 2>&1 | sed -u 's/^/[server]  /' &
(cd "$ROOT/client1" && exec "$PY" -u run_api.py) 2>&1 | sed -u 's/^/[client1] /' &
(cd "$ROOT/client2" && exec "$PY" -u run_api.py) 2>&1 | sed -u 's/^/[client2] /' &
sleep 1
"$PY" -u "$ROOT/phone_client/phone_bridge.py" 2>&1 | sed -u 's/^/[phone]   /' &

wait
