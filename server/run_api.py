#!/usr/bin/env python3
"""
Federated Learning Server API Runner
This starts only the Flask API server (no training logic)
The dashboard controls when to start actual training
"""

import sys
import os
from flask import Flask
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server_api import ServerAPI

def create_app():
    """Create and configure Flask app for server API"""
    app = Flask(__name__)
    CORS(app)
    
    server_api = ServerAPI(app)
    server_api.register_routes()
    server_api.log('Server API initialized')
    
    return app, server_api

if __name__ == '__main__':
    app, server_api = create_app()
    
    print("\n" + "="*60)
    print("Federated Learning Server API")
    print("="*60)
    print("Starting Server API on http://127.0.0.1:5000")
    print("Dashboard available at http://localhost:3000")
    print("="*60 + "\n")
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        server_api.log("Server API stopped")
        print("\n" + "="*60)
        print("Server API stopped")
        print("="*60)
