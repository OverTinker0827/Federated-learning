import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from model import Model
from client import Client
import threading
import os

NUM_CLIENTS = 5
LOCAL_EPOCHS = 10
ROUNDS = 10
BATCH_SIZE = 128
MAJOR_PERC = 0.8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


def partition_dataset_non_iid(num_clients=5, major_perc=0.8):
    labels = np.array(train_dataset.targets)

    client_indices = {i: [] for i in range(num_clients)}
    digits = np.arange(10)
    np.random.shuffle(digits)
    major_digits_per_client = 2
    digit_for_client = {}

    for cid in range(num_clients):
        major_digits = digits[cid * major_digits_per_client:(cid + 1) * major_digits_per_client]
        digit_for_client[cid] = major_digits

    for cid in range(num_clients):
        major_digits = digit_for_client[cid]
        major_idx = np.concatenate([np.where(labels == d)[0] for d in major_digits])
        np.random.shuffle(major_idx)
        n_major = int(len(major_idx) * major_perc)
        client_indices[cid].extend(major_idx[:n_major])

        other_digits = [d for d in digits if d not in major_digits]
        minor_idx = np.concatenate([np.where(labels == d)[0] for d in other_digits])
        np.random.shuffle(minor_idx)
        n_minor = int(len(minor_idx) * (1 - major_perc) / num_clients)
        client_indices[cid].extend(minor_idx[:n_minor])

        np.random.shuffle(client_indices[cid])

    return client_indices


partitions = partition_dataset_non_iid(NUM_CLIENTS, MAJOR_PERC)


global_model = Model().to(DEVICE)
if os.path.exists("weights.pth"):
    print("Loaded existing weights.")
    global_model.load_state_dict(torch.load("weights.pth", map_location=DEVICE))


clients = []
for cid, indices in partitions.items():
    loader = DataLoader(Subset(train_dataset, indices), batch_size=BATCH_SIZE, shuffle=True)
    clients.append(Client(client_id=cid, model=Model(), train_loader=loader, device=DEVICE))


for rnd in range(ROUNDS):
    print(f"\n--- Round {rnd + 1} ---")
    client_params = [None] * NUM_CLIENTS
    threads = []

    def train_client_thread(cid, client_obj):
        try:
            client_params[cid] = client_obj.train(global_model.state_dict(), epochs=LOCAL_EPOCHS)
        except Exception as e:
            print(f"Client {cid} failed: {e}")

    for cid, c in enumerate(clients):
        t = threading.Thread(target=train_client_thread, args=(cid, c))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    client_params = [cp for cp in client_params if cp is not None]
    if not client_params:

        continue


    new_state = {}
    for key in global_model.state_dict().keys():
        new_state[key] = sum([cp[key] for cp in client_params]) / len(client_params)
    global_model.load_state_dict(new_state)

    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in DataLoader(test_dataset, batch_size=128, shuffle=False):
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = global_model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Global model accuracy after round {rnd + 1}: {acc * 100:.2f}%")

torch.save(global_model.state_dict(), "weights.pth")

