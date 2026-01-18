import json
import os

CONFIG_FILE = 'client_config.json'
Client_IP="127.0.0.1"
class Config:
    """Client Configuration"""
    SERVER_IP = "127.0.0.1"
    SERVER_PORT = 8765
    CLIENT_ID = None
    CSV_PATH = "blood_bank_data_2.csv"
    
    @classmethod
    def load_from_file(cls):
        """Load configuration from JSON file if it exists"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                    cls.SERVER_IP = config_data.get('server_ip', cls.SERVER_IP)
                    cls.SERVER_PORT = config_data.get('server_port', cls.SERVER_PORT)
                    cls.CLIENT_ID = config_data.get('client_id', cls.CLIENT_ID)
                    cls.CSV_PATH = config_data.get('csv_path', cls.CSV_PATH)
            except Exception as e:
                print(f"Error loading config: {e}")
    
    @classmethod
    def save_to_file(cls):
        """Save configuration to JSON file"""
        try:
            config_data = {
                'server_ip': cls.SERVER_IP,
                'server_port': cls.SERVER_PORT,
                'client_id': cls.CLIENT_ID,
                'csv_path': cls.CSV_PATH
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            'server_ip': cls.SERVER_IP,
            'server_port': cls.SERVER_PORT,
            'client_id': cls.CLIENT_ID,
            'csv_path': cls.CSV_PATH
        }

# Load config on import
Config.load_from_file()