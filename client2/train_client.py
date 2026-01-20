import argparse
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from client_com import Client_Com
from model import Model
import torch.nn as nn
from config import Config


def load_preprocessed(pt_path):
    """Load preprocessed tensors from .pt file created by preprocessing.py"""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Preprocessed file not found: {pt_path}")
    
    data = torch.load(pt_path, weights_only=False)
    X_train = data["X_train"]
    y_train = data["y_train"]
    return X_train, y_train


def build_dataloader_from_tensors(X, y, batch_size=32, shuffle=True):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class Client:
    def __init__(self, device, train_loader, model, lr=1e-2, server_ip="127.0.0.1", port=8765, epochs=1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.comms = Client_Com(server_ip, port)
        self.training_complete = False

        try:
            self.client_id = self.comms.recieve_id()
        except Exception as e:
            print(f"Failed to initialize client: {e}")
            self.client_id = None
            raise

    def train(self, epochs=1):
        # Receive global weights
        try:
            global_weights = self.comms.recieve_weights()

            # None means either error or completion signal
            if global_weights is None:
                if not self.comms.connected:
                    print("Lost connection to server")
                    self.training_complete = True
                else:
                    print("Training complete signal received")
                    self.training_complete = True
                return False

            print("Received weights from server")
            self.model.load_state_dict(global_weights)
        except Exception as e:
            print(f"Failed to receive weights: {e}")
            self.training_complete = True
            return False

        self.model.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}/{epochs}")
            total_loss = 0
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("Detected NaN/Inf loss — stopping training")
                    try:
                        print(f"X any NaN: {torch.isnan(X).any().item()}, any Inf: {torch.isinf(X).any().item()}")
                        print(f"y any NaN: {torch.isnan(y).any().item()}, any Inf: {torch.isinf(y).any().item()}")
                        print(f"X min/max: {X.min().item() if X.numel() else 'NA'}/{X.max().item() if X.numel() else 'NA'}")
                        print(f"y min/max: {y.min().item()}/{y.max().item()}")
                        print(f"outputs min/max: {outputs.min().item()}/{outputs.max().item()}")
                    except Exception as e:
                        print(f"Diagnostics failed: {e}")
                    self.training_complete = True
                    return False

                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(self.train_loader))
            print(f"Client {self.client_id} | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # Send updated weights
        try:
            success = self.comms.send(self.model.state_dict())
            if success:
                print("Successfully sent weights to server")
                return True
            else:
                print("Failed to send weights")
                self.training_complete = True
                return False
        except Exception as e:
            print(f"Failed to send weights: {e}")
            self.training_complete = True
            return False

    def start(self):
        print(f"Client {self.client_id} starting training loop...")
        round_num = 0

        while not self.training_complete:
            try:
                round_num += 1
                print(f"\n=== Round {round_num} ===")
                success = self.train(epochs=self.epochs)

                if not success or self.training_complete:
                    break

            except KeyboardInterrupt:
                print("\nClient interrupted by user")
                break
            except Exception as e:
                print(f"Client {self.client_id} error: {e}")
                break

        print(f"\nClient {self.client_id} shutting down after {round_num} round(s)")
        self.comms.close()


def main():
    parser = argparse.ArgumentParser(description="Federated learning client using preprocessed data")
    parser.add_argument("index", nargs='?', default=None, help="index number used in preprocessed filename (e.g. 2 -> blood_bank_data_2_processed.pt)")
    parser.add_argument("--batch-size", type=int, default=32, help="local training batch size")
    parser.add_argument("--epochs", type=int, default=1, help="local training epochs per round")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="device to run training on")
    parser.add_argument("--client-id", type=int, default=None, help="client ID (used by API)")
    parser.add_argument("--server-ip", default=None, help="server IP address (used by API)")
    parser.add_argument("--server-port", type=int, default=None, help="server port (used by API)")
    args = parser.parse_args()

    idx = args.client_id
    if idx is None:
        print("Please provide a client ID (--client-id)")
        return
    
    # Load preprocessed .pt file created by preprocessing.py
    pt_path = f"blood_bank_data_{idx}_processed.pt"
    
    try:
        X_train, y_train = load_preprocessed(pt_path)
    except FileNotFoundError:
        print(f"Preprocessed file not found: {pt_path}")
        print("Please run preprocessing.py first to generate the preprocessed data.")
        return
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return

    print(f"Loaded preprocessed data from '{pt_path}'")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    device = torch.device(args.device)
    loader = build_dataloader_from_tensors(X_train, y_train, batch_size=args.batch_size, shuffle=True)

    model = Model(X_train.shape[2])

    # Use command-line args if provided, otherwise use Config defaults
    server_ip = args.server_ip if args.server_ip else Config.SERVER_IP
    server_port = args.server_port if args.server_port else Config.SERVER_PORT

    try:
        client = Client(device=device, train_loader=loader, model=model, lr=args.lr, 
                       server_ip=server_ip, port=server_port, epochs=args.epochs)
        client.start()
    except Exception as e:
        print(f"Client failed to start: {e}")


if __name__ == "__main__":
    main()