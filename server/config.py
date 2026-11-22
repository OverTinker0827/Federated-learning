import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server_config.json')


class Config:
    """Server Configuration"""
    HOST = "0.0.0.0"      # bind on all interfaces so the phone (via hotspot) can reach us
    PORT = 5000           # Flask control API port
    FL_PORT = 8765        # raw TCP port used by FL clients (see server.py)
    NUM_CLIENTS = 3
    ROUNDS = 5
    EPOCHS = 2
    RESUME = False        # start each demo from a fresh model so the learning curve is visible

    @classmethod
    def load_from_file(cls):
        """Load configuration from JSON file if it exists"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                    cls.HOST = config_data.get('host', cls.HOST)
                    cls.PORT = config_data.get('port', cls.PORT)
                    cls.NUM_CLIENTS = config_data.get('num_clients', cls.NUM_CLIENTS)
                    cls.ROUNDS = config_data.get('rounds', cls.ROUNDS)
                    cls.EPOCHS = config_data.get('epochs', cls.EPOCHS)
                    cls.RESUME = config_data.get('resume', cls.RESUME)
            except Exception as e:
                print(f"Error loading config: {e}")

    @classmethod
    def save_to_file(cls):
        """Save configuration to JSON file"""
        try:
            config_data = {
                'host': cls.HOST,
                'port': cls.PORT,
                'num_clients': cls.NUM_CLIENTS,
                'rounds': cls.ROUNDS,
                'epochs': cls.EPOCHS,
                'resume': cls.RESUME,
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            'host': cls.HOST,
            'port': cls.PORT,
            'fl_port': cls.FL_PORT,
            'num_clients': cls.NUM_CLIENTS,
            'rounds': cls.ROUNDS,
            'epochs': cls.EPOCHS,
            'resume': cls.RESUME,
        }

# Load config on import
Config.load_from_file()
