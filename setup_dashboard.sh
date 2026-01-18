#!/bin/bash

# 4. Install Dashboard Dependencies
echo "Installing Dashboard dependencies..."
cd dashboard
npm install
if [ $? -ne 0 ]; then
    echo "Failed to install dashboard dependencies"
    exit 1
fi

echo "Setup complete!"