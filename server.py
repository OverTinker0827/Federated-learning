# server_threaded.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from model import Model
from client import Client
import threading

NUM_CLIENTS = 5
LOCAL_EPOCHS = 1
ROUNDS = 5
BATCH_SIZE = 32
MAJOR_PERC = 0.8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Partition dataset (non-IID)
def partition_dataset_non_iid(num_clients=5, major_perc=0.8):
    labels = train_dataset.targets.numpy()
    client_indices = {i: [] for i in range(num_clients)}
    digits = np.arange(10)
    np.random.shuffle(digits)
    major_digits_per_client = 2
    digit_for_client = {}
    for cid in range(num_clients):
        major_digits = digits[cid*major_digits_per_client:(cid+1)*major_digits_per_client]
        digit_for_client[cid] = major_digits

    for cid in range(num_clients):
        major_digits = digit_for_client[cid]
        major_idx = np.concatenate([np.where(labels==d)[0] for d in major_digits])
        np.random.shuffle(major_idx)
        n_major = int(len(major_idx) * major_perc / num_clients)
        client_indices[cid].extend(major_idx[:n_major])

        other_digits = [d for d in digits if d not in major_digits]
        minor_idx = np.concatenate([np.where(labels==d)[0] for d in other_digits])
        np.random.shuffle(minor_idx)
        n_minor = int(len(minor_idx) * (1-major_perc) / num_clients)
        client_indices[cid].extend(minor_idx[:n_minor])

        np.random.shuffle(client_indices[cid])
    return client_indices

partitions = partition_dataset_non_iid(NUM_CLIENTS, MAJOR_PERC)

# Create clients
clients = []
for cid, indices in partitions.items():
    loader = DataLoader(Subset(train_dataset, indices), batch_size=BATCH_SIZE, shuffle=True)
    clients.append(Client(client_id=cid, model=Model(), train_loader=loader, device=DEVICE))

# Initialize global model
global_model = Model().to(DEVICE)

# ----------------------
# Federated training loop with threads
# ----------------------
for rnd in range(ROUNDS):
    print(f"\n--- Round {rnd+1} ---")
    client_params = [None] * NUM_CLIENTS  # thread-safe list to store results
    threads = []

    def train_client_thread(cid, client_obj):
        client_params[cid] = client_obj.train(global_model.state_dict(), epochs=LOCAL_EPOCHS)

    # Start threads
    for cid, c in enumerate(clients):
        t = threading.Thread(target=train_client_thread, args=(cid, c))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # FedAvg
    new_state = {}
    for key in global_model.state_dict().keys():
        new_state[key] = sum([cp[key] for cp in client_params]) / NUM_CLIENTS
    global_model.load_state_dict(new_state)

    # Evaluate global model
    global_model.eval()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = global_model(X)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Global model accuracy after round {rnd+1}: {acc*100:.2f}%")

# Optional: print client distributions
for c in clients:
    digits = [train_dataset.targets[i].item() for i in partitions[c.client_id]]
    unique, counts = np.unique(digits, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Client {c.client_id} distribution: {dist}")
