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
# Config.ROUNDS = 2
# Config.NUM_CLIENTS = 1
PORT=8765
# Config.HOST="127.0.0.1"
# global_model = Model().to(DEVICE)
# if os.path.exists("weights.pth"):
#     print("Loaded existing weights.")
#     global_model.load_state_dict(torch.load("weights.pth", map_location=DEVICE))



clients = {}          
client_count = 0
next_client_id=1
clients_lock = asyncio.Lock()
all_clients_connected = asyncio.Event()
stopped=asyncio.Event()
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
    except Exception:
        for c in clients:
            r,w =clients[c]
            if w==writer:
                clients.pop(c,None)
                break
        return None
    await writer.drain()


async def recv_weights(reader, device="cpu"):
    size_bytes = await recv_exact(reader, 8)
    size = struct.unpack("!Q", size_bytes)[0]
    try:
        payload = await recv_exact(reader, size)
    except Exception:
        for c in clients:
            r,w =clients[c]
            if r==reader:
                clients.pop(c,None)
                break
        return None
    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location=device)


async def handle_client(reader, writer):
    global client_count,next_client_id

    async with clients_lock:
        client_id = next_client_id
        next_client_id += 1
        clients[client_id] = (reader, writer)
        client_count+=1
    if client_count==Config.NUM_CLIENTS:
        all_clients_connected.set()

    addr = writer.get_extra_info("peername")
    print(f"Client {client_id} connected from {addr}")
    # fe.display(f"Client {client_id} connected from {addr}")
    # fe.display(f"Client count : {client_count}")
    # send client_id
    writer.write(struct.pack("!I", client_id))
    await writer.drain()

    try:
        # keep connection alive
        while True:
            await asyncio.sleep(500)
            # await reader.is_connected()
    except asyncio.CancelledError:
        pass
    finally:
        print(f"Client {client_id} disconnected")
        async with clients_lock:
            clients.pop(client_id, None)
            client_count-=1
            if client_count<Config.NUM_CLIENTS:
                all_clients_connected.clear()
        stopped.set()
        writer.close()
        await writer.wait_closed()

def federated_average(states):
    new_state = {}
    keys = states[0].keys()

    for k in keys:
        new_state[k] = sum(s[k] for s in states) / len(states)

    return new_state

async def broadcast_weights(state_dict):
    async with clients_lock:
        
        writers = [writer for _, writer in clients.values()]
    tasks=[]
    for writer in writers:
        temp=send_weights(writer, state_dict)
        if temp:
            tasks.append(temp)
        else:
            all_clients_connected.clear()
            stopped.set()


        
            
    # tasks = [send_weights(writer, state_dict) for writer in writers]
    if tasks:
        await asyncio.gather(*tasks,return_exceptions=True)


async def collect_updates(timeout=600):
    updates = {}

    async def recv_from_client(cid, reader):
        try:
            updates[cid] = await recv_weights(reader)
            if updates[cid] is None:
                del updates[cid]
                async with clients_lock:
                    clients.pop(cid, None)
                    client_count-=1

                    all_clients_connected.clear()
                    stopped.set()

            
        except Exception as e:
            print(f"Client {cid} failed: {e}")

    async with clients_lock:
        tasks = [
            recv_from_client(cid, reader)
            for cid, (reader, _) in clients.items()
        ]

    if tasks:
        await asyncio.wait(tasks, timeout=timeout)

    return updates

async def federated_training():
    global_model = Model(13).to(DEVICE)
    if os.path.exists("weights.pth"):
        print("Loaded existing weights.")
        global_model.load_state_dict(torch.load("weights.pth", map_location=DEVICE))
    # wait for at least one client
    # while True:
    #     async with clients_lock:
    #         if clients:
    #             break
    #     await asyncio.sleep(1)
    await all_clients_connected.wait()
    stopped.clear()
    round=0
    while True:
        round+=1
        
        if stopped.is_set():
            break
        try:
            # print(f"\n--- Round {round} ---")

            await broadcast_weights(global_model.state_dict())
            if stopped.is_set():
                break
            updates = await collect_updates()
            if stopped.is_set():
                break
            if not updates:
                # print("No updates received")
                continue

            new_state = federated_average(list(updates.values()))
            if stopped.is_set():
                break
            global_model.load_state_dict(new_state)
            if round==Config.ROUNDS:
                break

        except Exception:
            print("error occured")
            break

        print(f"Round {round + 1} aggregation complete")

    if round==Config.ROUNDS:
        torch.save(global_model.state_dict(), "global_weights.pth")
        print("Training finished")



async def main():
    server = await asyncio.start_server(handle_client, Config.HOST, PORT)
    print(f"Server listening on {PORT}")

    async with server:
        await asyncio.gather(
            server.serve_forever(),
            federated_training()
        )


if __name__ == "__main__":
    asyncio.run(main())

def start():
    asyncio.run(main())