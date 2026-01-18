"""
Flask API wrapper for Federated Learning Client
Manages client training via subprocess

Usage:
    from client_api import create_client_api_app
    
    client_id = 1
    flask_app, client_api, port = create_client_api_app(client_id)
    
    # Run Flask in background thread
    import threading
    api_thread = threading.Thread(
        target=lambda: flask_app.run(host='127.0.0.1', port=port, debug=False),
        daemon=True
    )
    api_thread.start()
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import subprocess
import os
import sys
from datetime import datetime
import threading
import time
import signal

class ClientAPI:
    def __init__(self, app=None, client_id=1):
        self.app = app
        self.client_id = client_id
        self.logs = []
        self.max_logs = 1000
        self.is_running = False
        self.training_process = None
        self.training_progress = 0
        self.is_training = False
        
        # Setup logging
        self.logger = logging.getLogger(f'ClientAPI-{client_id}')
        self.handler = logging.StreamHandler()
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        
        if app:
            self.init_app(app)

    def init_app(self, app):
        self.app = app
        CORS(app)
        self.register_routes()

    def register_routes(self):
        """Register all API routes"""
        self.app.route('/status', methods=['GET'])(self.get_status)
        self.app.route('/start', methods=['POST'])(self.start_client)
        self.app.route('/stop', methods=['POST'])(self.stop_client)
        self.app.route('/logs', methods=['GET'])(self.get_logs)
        self.app.route('/logs/clear', methods=['POST'])(self.clear_logs)

    def log(self, message, level='INFO'):
        """Add message to logs"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'[{timestamp}] [{level}] {message}'
        self.logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Also log to console
        if level == 'INFO':
            self.logger.info(message)
        elif level == 'WARNING':
            self.logger.warning(message)
        elif level == 'ERROR':
            self.logger.error(message)

    def get_status(self):
        """Get current client status"""
        try:
            return jsonify({
                'client_id': self.client_id,
                'running': self.is_running,
                'training': self.is_training,
                'progress': self.training_progress,
                'timestamp': datetime.now().isoformat()
            }), 200
        except Exception as e:
            self.log(f'Error getting status: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500

    def start_client(self):
        """Start the client training"""
        try:
            if self.is_running:
                return jsonify({'error': 'Client is already running'}), 400
            
            data = request.get_json() or {}
            server_ip = data.get('server_ip', '127.0.0.1')
            server_port = data.get('server_port', 8765)
            
            self.log(f'Starting client training, connecting to server at {server_ip}:{server_port}')
            
            # Start train_client.py as subprocess
            train_script = os.path.join(os.path.dirname(__file__), 'train_client.py')
            
            self.training_process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    train_script,
                    '--client-id', str(self.client_id),
                    '--server-ip', server_ip,
                    '--server-port', str(server_port),
                    ''
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.is_running = True
            self.log(f'Client training process started (PID: {self.training_process.pid})')
            
            # Read output in separate thread
            def read_output():
                try:
                    for line in self.training_process.stdout:
                        if line.strip():
                            self.log(line.strip())
                except:
                    pass
            
            output_thread = threading.Thread(target=read_output, daemon=True)
            output_thread.start()
            # Mark as training and monitor process in background
            self.is_training = True

            def monitor_process():
                try:
                    retcode = self.training_process.wait()
                    self.is_running = False
                    self.is_training = False
                    self.training_progress = 0
                    self.log(f'Client training process exited with code {retcode}')
                except Exception:
                    self.is_running = False
                    self.is_training = False

            monitor_thread = threading.Thread(target=monitor_process, daemon=True)
            monitor_thread.start()

            pid = self.training_process.pid if self.training_process else None
            return jsonify({
                'message': 'Client started',
                'client_id': self.client_id,
                'pid': pid
            }), 200
        except Exception as e:
            self.is_running = False
            self.log(f'Error starting client: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500

    def stop_client(self):
        """Stop the client training"""
        try:
            if not self.is_running:
                return jsonify({'error': 'Client is not running'}), 400
            # Try to terminate the training process if it exists
            try:
                if self.training_process and self.training_process.poll() is None:
                    if hasattr(signal, 'CTRL_C_EVENT'):
                        # Windows
                        self.training_process.send_signal(signal.CTRL_C_EVENT)
                    else:
                        os.kill(self.training_process.pid, signal.SIGTERM)
                    try:
                        self.training_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.training_process.kill()
                        self.training_process.wait()
            except Exception as e:
                self.log(f'Error terminating process: {e}', 'WARNING')

            self.is_running = False
            self.is_training = False
            self.training_progress = 0
            self.training_process = None
            self.log('Client stopped')

            return jsonify({'message': 'Client stopped'}), 200
        except Exception as e:
            self.log(f'Error stopping client: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500

    def get_logs(self):
        """Get client logs"""
        try:
            offset = request.args.get('offset', 0, type=int)
            logs = self.logs[offset:] if offset < len(self.logs) else []
            
            return jsonify({
                'logs': '\n'.join(logs),
                'total_lines': len(self.logs),
                'returned_lines': len(logs)
            }), 200
        except Exception as e:
            self.log(f'Error getting logs: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500

    def clear_logs(self):
        """Clear all client logs"""
        try:
            self.logs = []
            self.log('Logs cleared')
            return jsonify({'message': 'Logs cleared'}), 200
        except Exception as e:
            self.log(f'Error clearing logs: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500

    def update_progress(self, progress):
        """Update training progress (0-100)"""
        self.training_progress = min(100, max(0, progress))


def create_client_api_app(client_id=1):
    """Create and configure the Flask app for client API"""
    app = Flask(__name__)
    CORS(app)
    
    # Determine port based on client ID
    port = 6000 + client_id
    
    client_api = ClientAPI(app, client_id=client_id)
    
    return app, client_api, port


# Example usage in your train_client.py:
"""
from client_api import create_client_api_app

# In your main code:
client_id = 1
flask_app, client_api, port = create_client_api_app(client_id)

# When you log something:
client_api.log('Your log message here')

# When you update progress:
client_api.update_progress(50)  # 50% complete

# Run Flask app in a separate thread:
import threading
flask_thread = threading.Thread(
    target=lambda: flask_app.run(host='127.0.0.1', port=port, debug=False),
    daemon=True
)
flask_thread.start()
"""
