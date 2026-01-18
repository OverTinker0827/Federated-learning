#!/bin/bash

# 1. Update system and install base tools (Node, npm, Python venv)
echo "Updating system and installing base dependencies..."
sudo apt-get update
sudo apt-get install -y nodejs npm python3-pip python3-venv

# 2. Setup Python Virtual Environment
echo "Setting up Python Virtual Environment..."
# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# ACTIVATE IT (This is crucial)
source .venv/bin/activate

# 3. Install Python Dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install Python dependencies"
    exit 1
fi

