#!/bin/bash

# Federated Learning Dashboard - Quick Start Script

echo "=========================================="
echo "Federated Learning Dashboard Setup"
echo "=========================================="
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js v14 or higher."
    exit 1
fi

echo "✓ Node.js $(node -v) detected"
echo "✓ npm $(npm -v) detected"
echo ""

# Install dashboard dependencies
echo "📦 Installing dashboard dependencies..."
cd dashboard
npm install

if [ $? -eq 0 ]; then
    echo "✓ Dashboard dependencies installed"
else
    echo "❌ Failed to install dashboard dependencies"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Install Python dependencies (on your server/client machines):"
echo "   pip install flask flask-cors"
echo ""
echo "2. Integrate the API files:"
echo "   - Copy server/server_api.py to your server"
echo "   - Copy client/client_api.py to your client"
echo ""
echo "3. Update server_config.json to set num_clients"
echo ""
echo "4. Start the servers in separate terminals:"
echo "   Terminal 1: python server/server.py"
echo "   Terminal 2: python client/train_client.py (repeat for each client)"
echo ""
echo "5. Start the dashboard:"
echo "   cd dashboard && npm start"
echo ""
echo "6. Open browser to http://localhost:3000"
echo ""
echo "📚 For detailed setup instructions, see: DASHBOARD_SETUP.md"
echo ""
