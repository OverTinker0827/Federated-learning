import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from model import Model
from client import Client
import threading
import os
from config import Config
# Config.NUM_CLIENTS = 5
LOCAL_EPOCHS = 1
# ROUNDS = 1
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


partitions = partition_dataset_non_iid(Config.NUM_CLIENTS, MAJOR_PERC)




for cid, indices in partitions.items():
    loader = DataLoader(Subset(train_dataset, indices), batch_size=BATCH_SIZE, shuffle=True)

    client1=Client( model=Model(), train_loader=loader, device=DEVICE)
    client1.start()
    break
    

