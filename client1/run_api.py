#!/usr/bin/env python3
"""
Federated Learning Client API Runner
This starts only the Flask API for a specific client
The dashboard controls when to start actual training
"""

import sys
import os
import argparse
from flask import Flask
from flask_cors import CORS
from config import Client_IP
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client_api import ClientAPI

def create_app(client_id=1):
    """Create and configure Flask app for client API"""
    app = Flask(__name__)
    CORS(app)
    
    client_api = ClientAPI(app, client_id=client_id)
    client_api.register_routes()
    client_api.log(f'Client {client_id} API initialized')
    
    return app, client_api

if __name__ == '__main__':

    ip=Client_IP
    client_id = 1
    port = 6001
    app, client_api = create_app(client_id)
    
    print("\n" + "="*60)
    print(f"Federated Learning Client {client_id} API")
    print("="*60)
    print(f"Starting Client {client_id} API on http://{ip}:{port}")
    print("Dashboard available at http://localhost:3000")
    print("="*60 + "\n")
    
    try:
        app.run(host=ip, port=port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        client_api.log(f"Client {client_id} API stopped")
        print(f"\n" + "="*60)
        print(f"Client {client_id} API stopped")
        print("="*60)
