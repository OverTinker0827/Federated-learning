"""
Flask API wrapper for the Federated Learning Server
Manages server startup/shutdown via subprocess

Usage:
    from server_api import create_server_api_app
    
    flask_app, server_api = create_server_api_app()
    
    # Run Flask in background thread
    import threading
    api_thread = threading.Thread(
        target=lambda: flask_app.run(host='127.0.0.1', port=5000, debug=False),
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
from config import Config
import threading
import time
import signal

class ServerAPI:
    def __init__(self, app=None):
        self.app = app
        self.logs = []
        self.max_logs = 1000
        self.is_running = False
        self.server_process = None
        self.log_file = None
        
        # Setup logging
        self.logger = logging.getLogger('ServerAPI')
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
        self.app.route('/config', methods=['GET'])(self.get_config)
        self.app.route('/config/update', methods=['POST'])(self.update_config)
        self.app.route('/start', methods=['POST'])(self.start_server)
        self.app.route('/stop', methods=['POST'])(self.stop_server)
        self.app.route('/status', methods=['GET'])(self.get_status)
        self.app.route('/logs', methods=['GET'])(self.get_logs)
        self.app.route('/metrics', methods=['GET'])(self.get_metrics)
        self.app.route('/logs/clear', methods=['POST'])(self.clear_logs)

    def get_metrics(self):
        """Return evaluation metrics computed by the training server (if any)"""
        try:
            results_path = os.path.join(os.path.dirname(__file__), 'test_results.json')
            if not os.path.exists(results_path):
                return jsonify({'error': 'No metrics available'}), 404
            import json
            with open(results_path, 'r') as fh:
                data = json.load(fh)
            return jsonify({'metrics': data}), 200
        except Exception as e:
            self.log(f'Error getting metrics: {e}', 'ERROR')
            return jsonify({'error': str(e)}), 500

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

    def get_config(self):
        """Get current server configuration"""
        try:
            return jsonify(Config.to_dict()), 200
        except Exception as e:
            self.log(f'Error getting config: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500

    def update_config(self):
        """Update server configuration"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Update config
            if 'host' in data:
                Config.HOST = data['host']
            if 'port' in data:
                Config.PORT = data['port']
            if 'num_clients' in data:
                Config.NUM_CLIENTS = data['num_clients']
            if 'rounds' in data:
                Config.ROUNDS = data['rounds']
            if 'epochs' in data:
                Config.EPOCHS = data['epochs']
            
            # Save to file
            Config.save_to_file()
            self.log(f'Configuration updated: {data}')
            
            return jsonify({
                'message': 'Configuration updated successfully',
                'config': Config.to_dict()
            }), 200
        except Exception as e:
            self.log(f'Error updating config: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500

    def start_server(self):
        """Start the training server"""
        try:
            if self.is_running:
                return jsonify({'error': 'Server is already running'}), 400
            
            self.is_running = True
            self.log(f'Server starting with {Config.NUM_CLIENTS} clients')
            
            # Start server in a separate thread
            server_thread = threading.Thread(target=self._run_server, daemon=True)
            server_thread.start()
            
            return jsonify({
                'message': 'Server started',
                'config': Config.to_dict()
            }), 200
        except Exception as e:
            self.is_running = False
            self.log(f'Error starting server: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500

    def _run_server(self):
        """Start the actual server as a subprocess"""
        try:
            server_script = os.path.join(os.path.dirname(__file__), 'server.py')

            # Start server.py as subprocess with unbuffered output
            self.server_process = subprocess.Popen(
                [sys.executable, "-u", server_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            self.log(f'Server process started with PID: {self.server_process.pid}')

            # Read combined stdout/stderr and log lines
            def read_output():
                try:
                    for line in self.server_process.stdout:
                        if line and line.strip():
                            self.log(line.rstrip())
                except Exception:
                    pass

            output_thread = threading.Thread(target=read_output, daemon=True)
            output_thread.start()

            # Wait for process to complete
            retcode = self.server_process.wait()
            self.log(f'Server process exited with code {retcode}')
            self.is_running = False
            self.server_process = None

        except Exception as e:
            self.log(f'Failed to start server process: {str(e)}', 'ERROR')
            self.is_running = False
            self.server_process = None

    def stop_server(self):
        """Stop the training server"""
        try:
            if not self.is_running and not self.server_process:
                return jsonify({'error': 'Server is not running'}), 400

            # Try to terminate subprocess if running
            try:
                if self.server_process and self.server_process.poll() is None:
                    if hasattr(signal, 'CTRL_C_EVENT'):
                        self.server_process.send_signal(signal.CTRL_C_EVENT)
                    else:
                        os.kill(self.server_process.pid, signal.SIGTERM)
                    try:
                        self.server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.server_process.kill()
                        self.server_process.wait()
            except Exception as e:
                self.log(f'Error terminating server process: {e}', 'WARNING')

            self.is_running = False
            self.server_process = None
            self.log('Server stopped')

            return jsonify({'message': 'Server stopped'}), 200
        except Exception as e:
            self.log(f'Error stopping server: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500

    def get_status(self):
        """Get current server status"""
        try:
            return jsonify({
                'running': self.is_running,
                'timestamp': datetime.now().isoformat(),
                'config': Config.to_dict(),
                'log_count': len(self.logs)
            }), 200
        except Exception as e:
            self.log(f'Error getting status: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500

    def get_logs(self):
        """Get server logs"""
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
        """Clear all server logs"""
        try:
            self.logs = []
            self.log('Logs cleared')
            return jsonify({'message': 'Logs cleared'}), 200
        except Exception as e:
            self.log(f'Error clearing logs: {str(e)}', 'ERROR')
            return jsonify({'error': str(e)}), 500


def create_server_api_app():
    """Create and configure the Flask app for server API"""
    app = Flask(__name__)
    CORS(app)
    
    server_api = ServerAPI(app)
    
    return app, server_api


# Example usage in your server.py:
"""
from server_api import create_server_api_app

# In your main code:
flask_app, server_api = create_server_api_app()

# When you log something:
server_api.log('Your log message here')

# Run Flask app in a separate thread:
import threading
flask_thread = threading.Thread(
    target=lambda: flask_app.run(host='127.0.0.1', port=5000, debug=False),
    daemon=True
)
flask_thread.start()
"""
