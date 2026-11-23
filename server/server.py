import torch
import numpy as np
from model import Model, load_arch, describe
import threading
import os
import asyncio
import struct
import io
import json
from config import Config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PORT=8765
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "global_weights.pth")
TEST_DATA_PATH = os.path.join(HERE, "test_data.pt")
TEST_RESULTS_PATH = os.path.join(HERE, "test_results.json")
ROUND_METRICS_PATH = os.path.join(HERE, "round_metrics.json")
MODEL_CONFIG_PATH = os.path.join(HERE, "model_config.json")      # written by the dashboard Model Builder
GLOBAL_ARCH_PATH = os.path.join(HERE, "global_model_arch.json")  # architecture of global_weights.pth

clients = {}          
client_count = 0
next_client_id=1
clients_lock = asyncio.Lock()
all_clients_connected = asyncio.Event()
stopped=asyncio.Event()
training_complete=asyncio.Event()

async def recv_exact(reader, size):
    data = b""
    while len(data) < size:
        chunk = await reader.read(size - len(data))
        if not chunk:
            raise ConnectionError("Client disconnected")
        data += chunk
    return data

async def send_weights(writer, state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    payload = buffer.getvalue()
    try:
        writer.write(struct.pack("!Q", len(payload)))
        writer.write(payload)
        await writer.drain()
        return True
    except Exception as e:
        print(f"Failed to send weights: {e}")
        return False

async def send_completion_signal(writer):
    """Send signal to client that training is complete"""
    try:
        # Send special signal: size = 0 means training complete
        writer.write(struct.pack("!Q", 0))
        await writer.drain()
        return True
    except Exception:
        return False

async def recv_weights(reader, device="cpu"):
    try:
        size_bytes = await recv_exact(reader, 8)
        size = struct.unpack("!Q", size_bytes)[0]
        payload = await recv_exact(reader, size)
        buffer = io.BytesIO(payload)
        return torch.load(buffer, map_location=device)
    except ConnectionError:
        return None
    except Exception as e:
        print(f"Error receiving weights: {e}")
        return None

async def handle_client(reader, writer):
    global client_count, next_client_id

    async with clients_lock:
        client_id = next_client_id
        next_client_id += 1
        clients[client_id] = (reader, writer)
        client_count += 1

    if client_count == Config.NUM_CLIENTS:
        all_clients_connected.set()

    addr = writer.get_extra_info("peername")
    print(f"Client {client_id} connected from {addr}")

    # Send client_id
    try:
        writer.write(struct.pack("!I", client_id))
        await writer.drain()
    except Exception as e:
        print(f"Failed to send client ID: {e}")
        async with clients_lock:
            clients.pop(client_id, None)
            client_count -= 1
        return

    try:
        # Keep connection alive until training is complete
        while not training_complete.is_set():
            await asyncio.sleep(1)

        # Notify client that training is done
        await send_completion_signal(writer)
        await asyncio.sleep(0.5)  # Give client time to receive signal

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Client {client_id} error: {e}")
    finally:
        print(f"Client {client_id} disconnected")
        async with clients_lock:
            clients.pop(client_id, None)
            client_count -= 1
            if client_count < Config.NUM_CLIENTS and not training_complete.is_set():
                all_clients_connected.clear()
                stopped.set()

        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

def federated_average(states):
    new_state = {}
    keys = states[0].keys()
    for k in keys:
        new_state[k] = sum(s[k] for s in states) / len(states)
    return new_state

async def broadcast_weights(state_dict):
    async with clients_lock:
        client_list = list(clients.items())

    failed_clients = []
    tasks = []

    for cid, (_, writer) in client_list:
        task = send_weights(writer, state_dict)
        tasks.append((cid, task))

    if tasks:
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
        for (cid, _), result in zip(tasks, results):
            if isinstance(result, Exception) or result is False:
                failed_clients.append(cid)

    # Remove failed clients
    if failed_clients:
        async with clients_lock:
            for cid in failed_clients:
                clients.pop(cid, None)
        print(f"Removed {len(failed_clients)} failed client(s)")
        if len(clients) == 0:
            stopped.set()

async def collect_updates(timeout=600):
    updates = {}

    async def recv_from_client(cid, reader):
        try:
            weight = await recv_weights(reader)
            if weight is not None:
                updates[cid] = weight
            else:
                async with clients_lock:
                    clients.pop(cid, None)
        except Exception as e:
            print(f"Client {cid} failed during collection: {e}")
            async with clients_lock:
                clients.pop(cid, None)

    async with clients_lock:
        tasks = [
            asyncio.create_task(recv_from_client(cid, reader))   # ✅ FIX
            for cid, (reader, _) in clients.items()
        ]

    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=timeout)

        # Optional cleanup
        for task in pending:
            task.cancel()

    return updates


def evaluate(model):
    """Evaluate the global model on the held-out marketplace test set (in units)."""
    if not os.path.exists(TEST_DATA_PATH):
        return None
    data = torch.load(TEST_DATA_PATH, map_location=DEVICE, weights_only=False)
    X_test, y_test = data['X_test'], data['y_test']
    meta = data.get('metadata', {})
    y_mean, y_std = float(meta.get('y_mean', 0.0)), float(meta.get('y_std', 1.0))
    if len(X_test) == 0:
        return None
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test.to(DEVICE)).cpu().numpy()
    y_true = y_test.cpu().numpy() * y_std + y_mean
    y_pred = np.clip(preds_scaled * y_std + y_mean, 0, None)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        'mse': mse,
        'mae': mae,
        'rmse': float(np.sqrt(mse)),
        'r2': 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0,
        'num_samples': int(len(y_true)),
        'feature_dim': int(X_test.shape[1]),
        'target': 'units sold in next 7 days',
        '_y_true': y_true,
        '_y_pred': y_pred,
    }


def write_round_metrics(history, status):
    try:
        with open(ROUND_METRICS_PATH, 'w') as fh:
            json.dump({'status': status, 'rounds_total': Config.ROUNDS, 'history': history}, fh, indent=2)
    except Exception as e:
        print(f"Could not write round metrics: {e}")


async def federated_training():
    arch = load_arch(MODEL_CONFIG_PATH)
    global_model = Model(arch=arch).to(DEVICE)
    arch = global_model.arch
    print(f"Model architecture: {describe(arch)}")
    if Config.RESUME and os.path.exists(WEIGHTS_PATH):
        try:
            state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            global_model.load_state_dict(state)
            print("Loaded existing global weights (resume enabled).")
        except Exception as e:
            print(f"Could not load existing weights (shape mismatch or corrupt file): {e}. Starting fresh.")
    else:
        print("Initialised fresh global DemandNet model "
              f"({sum(p.numel() for p in global_model.parameters())} parameters).")

    # Clear results of a previous demo run so the dashboard doesn't show stale metrics
    for stale in (TEST_RESULTS_PATH, ROUND_METRICS_PATH):
        if os.path.exists(stale):
            os.remove(stale)

    history = []
    base = evaluate(global_model)
    if base:
        history.append({'round': 0, 'mae': base['mae'], 'rmse': base['rmse'], 'r2': base['r2'], 'clients': 0})
        print(f"Round 0 (untrained) | Test MAE: {base['mae']:.2f} units | R2: {base['r2']:.3f}")
    write_round_metrics(history, 'waiting_for_clients')

    print(f"Waiting for all {Config.NUM_CLIENTS} clients to connect...")
    await all_clients_connected.wait()
    print(f"All {Config.NUM_CLIENTS} client(s) connected. Starting training.")
    write_round_metrics(history, 'training')

    stopped.clear()
    round_num = 0

    while round_num < Config.ROUNDS:
        if stopped.is_set():
            print("Training stopped due to client disconnection")
            break

        round_num += 1
        print(f"\n--- Round {round_num}/{Config.ROUNDS} ---")

        try:
            print(f"Broadcasting global model to {len(clients)} client(s)")
            # the architecture travels with the weights so every client builds the same network
            await broadcast_weights({"arch": arch, "state": global_model.state_dict()})
            if stopped.is_set():
                break

            updates = await collect_updates()
            if stopped.is_set():
                break

            if not updates:
                print("No updates received")
                stopped.set()
                break

            print(f"Received updates from clients {sorted(updates.keys())}; running FedAvg")
            new_state = federated_average(list(updates.values()))
            global_model.load_state_dict(new_state)

            m = evaluate(global_model)
            if m:
                history.append({'round': round_num, 'mae': m['mae'], 'rmse': m['rmse'], 'r2': m['r2'],
                                'clients': len(updates)})
                print(f"Round {round_num} aggregation complete | Test MAE: {m['mae']:.2f} units | "
                      f"RMSE: {m['rmse']:.2f} | R2: {m['r2']:.3f}")
            else:
                print(f"Round {round_num} aggregation complete")
            write_round_metrics(history, 'training')

        except Exception as e:
            print(f"Error during training round: {e}")
            stopped.set()
            break

    if round_num == Config.ROUNDS and not stopped.is_set():
        torch.save(global_model.state_dict(), WEIGHTS_PATH)
        with open(GLOBAL_ARCH_PATH, 'w') as fh:
            json.dump(arch, fh, indent=2)
        print(f"\nTraining completed successfully after {Config.ROUNDS} rounds!")
        print(f"Saved global model to {WEIGHTS_PATH}")

    try:
        m = evaluate(global_model)
        if m:
            y_true, y_pred = m.pop('_y_true'), m.pop('_y_pred')
            n_show = min(20, len(y_true))
            m['sample_predictions'] = [
                {'actual': round(float(y_true[i]), 2), 'predicted': round(float(y_pred[i]), 2),
                 'error': round(float(abs(y_true[i] - y_pred[i])), 2)}
                for i in range(n_show)
            ]
            m['prediction_stats'] = {
                'y_true_min': float(np.min(y_true)), 'y_true_max': float(np.max(y_true)),
                'y_true_mean': float(np.mean(y_true)), 'y_pred_min': float(np.min(y_pred)),
                'y_pred_max': float(np.max(y_pred)), 'y_pred_mean': float(np.mean(y_pred)),
            }
            m['round_history'] = history
            with open(TEST_RESULTS_PATH, 'w') as fh:
                json.dump(m, fh, indent=2)
            print(f"Evaluation results: MSE={m['mse']:.4f}, MAE={m['mae']:.4f}, "
                  f"RMSE={m['rmse']:.4f}, R2={m['r2']:.4f}")
        else:
            print('No test_data.pt found; skipping evaluation')
    except Exception as e:
        print('Error during evaluation:', e)

    write_round_metrics(history, 'complete' if round_num == Config.ROUNDS and not stopped.is_set() else 'stopped')
    training_complete.set()

    # Give time for completion signals to be sent
    await asyncio.sleep(2)

async def main():
    server = await asyncio.start_server(handle_client, Config.HOST, PORT)
    print(f"Server listening on {Config.HOST}:{PORT}")

    async with server:
        training_task = asyncio.create_task(federated_training())
        server_task = asyncio.create_task(server.serve_forever())

        # Wait for training to complete
        await training_task

        # Cancel server after training is done
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

        print("Server shutting down.")
        exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer interrupted by user")

def start():
    asyncio.run(main())