# model.py
"""
Configurable MLP for demand forecasting (predicts standardised units sold in
the next 7 days).

The architecture is a small JSON description shared by every participant:

    {"name": "...", "input_dim": 16,
     "layers": [{"units": 64, "activation": "relu"}, ..., {"units": 1, "activation": "linear"}]}

The last layer is the output (1 unit). Layers are named fc1, fc2, ... so the
default (Preset 1, DemandNet 16->64->32->1 ReLU) keeps its original parameter
names. The same description drives the JavaScript trainer on the phone
(phone_client/app/trainer.js), so names, shapes and activations must match.
"""
import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_FEATURES = 16

ACTIVATIONS = {
    "relu": torch.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, 0.01),
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "linear": lambda x: x,
}
MAX_HIDDEN_LAYERS = 6
MAX_UNITS = 256

PRESET_1 = {
    "name": "Preset 1 - DemandNet",
    "input_dim": NUM_FEATURES,
    "layers": [
        {"units": 64, "activation": "relu"},
        {"units": 32, "activation": "relu"},
        {"units": 1, "activation": "linear"},
    ],
}
PRESETS = {"preset1": PRESET_1}


def validate_arch(arch):
    """Return a clean copy of arch, or raise ValueError."""
    if not isinstance(arch, dict) or not isinstance(arch.get("layers"), list):
        raise ValueError("architecture must have a 'layers' list")
    layers = arch["layers"]
    if not 1 <= len(layers) <= MAX_HIDDEN_LAYERS + 1:
        raise ValueError(f"use 0-{MAX_HIDDEN_LAYERS} hidden layers")
    clean = []
    for i, layer in enumerate(layers):
        units = int(layer.get("units", 0))
        act = str(layer.get("activation", "relu"))
        if act not in ACTIVATIONS:
            raise ValueError(f"layer {i + 1}: unknown activation '{act}'")
        if not 1 <= units <= MAX_UNITS:
            raise ValueError(f"layer {i + 1}: units must be 1-{MAX_UNITS}")
        clean.append({"units": units, "activation": act})
    if clean[-1]["units"] != 1:
        raise ValueError("output layer must have exactly 1 unit")
    return {
        "name": str(arch.get("name") or "Custom MLP")[:60],
        "input_dim": int(arch.get("input_dim", NUM_FEATURES)),
        "layers": clean,
    }


def load_arch(path):
    """Architecture saved by the dashboard Model Builder, else Preset 1."""
    try:
        with open(path) as fh:
            return validate_arch(json.load(fh))
    except (OSError, ValueError, json.JSONDecodeError):
        return copy.deepcopy(PRESET_1)


def describe(arch):
    sizes = [arch["input_dim"]] + [l["units"] for l in arch["layers"]]
    acts = ",".join(l["activation"] for l in arch["layers"][:-1]) or "none"
    return f"{arch.get('name', 'MLP')}: {'->'.join(map(str, sizes))} ({acts})"


class Model(nn.Module):
    def __init__(self, input_dim=None, arch=None):
        super().__init__()
        arch = validate_arch(arch or PRESET_1)
        if input_dim is not None:
            arch["input_dim"] = int(input_dim)
        self.arch = arch
        prev = arch["input_dim"]
        self._acts = []
        for i, layer in enumerate(arch["layers"]):
            setattr(self, f"fc{i + 1}", nn.Linear(prev, layer["units"]))
            self._acts.append(ACTIVATIONS[layer["activation"]])
            prev = layer["units"]

    def forward(self, x):
        for i, act in enumerate(self._acts):
            x = act(getattr(self, f"fc{i + 1}")(x))
        return x.squeeze(-1)
