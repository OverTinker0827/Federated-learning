import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from model import Model
import threading
import os
import asyncio
import struct
import io
from config import Config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PORT=8765

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


async def federated_training():
    global_model = Model(13).to(DEVICE)
    if os.path.exists("global_weights.pth"):
        print("Loaded existing weights.")
        global_model.load_state_dict(torch.load("global_weights.pth", map_location=DEVICE))

    print("Waiting for all clients to connect...")
    await all_clients_connected.wait()
    print(f"All {Config.NUM_CLIENTS} client(s) connected. Starting training.")

    stopped.clear()
    round_num = 0

    while round_num < Config.ROUNDS:
        if stopped.is_set():
            print("Training stopped due to client disconnection")
            break

        round_num += 1
        print(f"\n--- Round {round_num}/{Config.ROUNDS} ---")

        try:
            # Broadcast weights
            await broadcast_weights(global_model.state_dict())
            if stopped.is_set():
                break

            # Collect updates
            updates = await collect_updates()
            if stopped.is_set():
                break

            if not updates:
                print("No updates received")
                stopped.set()
                break

            # Aggregate
            new_state = federated_average(list(updates.values()))
            global_model.load_state_dict(new_state)

            print(f"Round {round_num} aggregation complete")

        except Exception as e:
            print(f"Error during training round: {e}")
            stopped.set()
            break

    if round_num == Config.ROUNDS:
        torch.save(global_model.state_dict(), "global_weights.pth")
        print(f"\nTraining completed successfully after {Config.ROUNDS} rounds!")
        print("saving values")

    # After training completes (or stops), evaluate on test dataset if available
    try:
        test_path = os.path.join(os.path.dirname(__file__), 'test.csv')
        if os.path.exists(test_path):
            print("Running evaluation on test dataset...")
            try:
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                import json

                def create_sequences(data, target, seq_len=7):
                    X, y = [], []
                    for i in range(len(data) - seq_len):
                        X.append(data[i:i + seq_len])
                        y.append(target[i + seq_len])
                    return np.array(X, dtype=float), np.array(y, dtype=float)

                df = pd.read_csv(test_path)
                if 'Units_Used_tomorrow' in df.columns:
                    # Encode categorical
                    if 'Blood_Type' in df.columns:
                        le = LabelEncoder()
                        df['Blood_Type'] = le.fit_transform(df['Blood_Type'].astype(str))

                    if 'Date' in df.columns:
                        df = df.sort_values('Date')

                    target = pd.to_numeric(df['Units_Used_tomorrow'], errors='coerce')
                    target = target.fillna(method='ffill').fillna(method='bfill')
                    target = target.values.astype(float)

                    features = df.drop(columns=[c for c in ['Date', 'Units_Used_tomorrow'] if c in df.columns])
                    features = features.apply(pd.to_numeric, errors='coerce')
                    features = features.dropna(axis=1, how='all')
                    non_constant_cols = features.columns[features.nunique() > 1]
                    features = features[non_constant_cols]
                    features = features.fillna(features.mean())

                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)

                    X_seq, y_seq = create_sequences(features_scaled, target, seq_len=7)

                    if X_seq.size == 0:
                        print('No sequences generated from test set; skipping evaluation')
                    else:
                        device = DEVICE
                        global_model.to(device)
                        global_model.eval()
                        import torch as _torch
                        with _torch.no_grad():
                            X_tensor = _torch.tensor(X_seq, dtype=_torch.float32, device=device)
                            preds = global_model(X_tensor).cpu().numpy()

                        y_true = y_seq
                        y_pred = preds

                        # Compute metrics
                        mse = float(np.mean((y_true - y_pred) ** 2))
                        mae = float(np.mean(np.abs(y_true - y_pred)))
                        rmse = float(np.sqrt(mse))
                        ss_res = float(np.sum((y_true - y_pred) ** 2))
                        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

                        results = {
                            'mse': mse,
                            'mae': mae,
                            'rmse': rmse,
                            'r2': r2,
                            'num_samples': int(len(y_true))
                        }

                        # Save results
                        results_path = os.path.join(os.path.dirname(__file__), 'test_results.json')
                        with open(results_path, 'w') as fh:
                            json.dump(results, fh)

                        print('Evaluation results saved to', results_path)
                else:
                    print("Test CSV does not contain 'Units_Used_tomorrow' target; skipping evaluation")
            except Exception as e:
                print('Error during evaluation:', e)
        else:
            print('No test.csv found; skipping evaluation')
    except Exception:
        pass

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