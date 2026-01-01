import socket
import struct
import torch
import io

class Client_Com:
    def __init__(self, server_ip, port):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        try:
            self.client.connect((server_ip, port))
            self.client.settimeout(300)  # 30 second timeout
            self.connected = True
            print(f"Connected to server at {server_ip}:{port}")
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            raise

    def send(self, state_dict):
        if not self.connected:
            return False
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        weights_in_bytes = buffer.getvalue()
        try:
            self.client.sendall(struct.pack("!Q", len(weights_in_bytes)))
            self.client.sendall(weights_in_bytes)  
            return True
        except (socket.error, BrokenPipeError, ConnectionResetError) as e:
            print(f"Failed to send weights: {e}")
            self.connected = False
            return False

    def recv_with_size(self, n):
        data = b""
        while len(data) < n:
            try:
                chunk = self.client.recv(n - len(data))
                if not chunk:
                    raise ConnectionError("Server disconnected")
                data += chunk
            except socket.timeout:
                raise ConnectionError("Connection timed out")
            except (socket.error, ConnectionResetError) as e:
                raise ConnectionError(f"Connection lost: {e}")
        return data

    def recieve_id(self):
        try:
            msg = self.recv_with_size(4)
            client_id = struct.unpack("!I", msg)[0]
            print(f"Received client ID: {client_id}")
            return client_id
        except Exception as e:
            print(f"Failed to receive client ID: {e}")
            self.connected = False
            raise

    def recieve_weights(self):
        try:
            size_bytes = self.recv_with_size(8)
            size = struct.unpack("!Q", size_bytes)[0]

            # Size 0 is a special signal meaning training is complete
            if size == 0:
                print("Received completion signal from server")
                return None

            weight_in_bytes = self.recv_with_size(size)
            buffer = io.BytesIO(weight_in_bytes)
            state_dict = torch.load(buffer, map_location="cpu")
            return state_dict
        except ConnectionError as e:
            print(f"Connection error while receiving weights: {e}")
            self.connected = False
            return None
        except Exception as e:
            print(f"Failed to receive weights: {e}")
            self.connected = False
            return None

    def close(self):
        try:
            self.client.close()
            self.connected = False
            print("Connection closed")
        except Exception:
            pass