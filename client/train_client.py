import argparse
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from client_com import Client_Com
from model import Model
import torch.nn as nn



def create_sequences(data, target, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(target[i + seq_len])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_and_process(csv_path, seq_len=7):
    df = pd.read_csv(csv_path)

    # encode categorical and sort by date
    if "Blood_Type" in df.columns:
        le = LabelEncoder()
        df["Blood_Type"] = le.fit_transform(df["Blood_Type"])

    if "Date" in df.columns:
        df = df.sort_values("Date")

    # expect target column named 'Units_Used_tomorrow'
    if "Units_Used_tomorrow" not in df.columns:
        raise ValueError("CSV must contain 'Units_Used_tomorrow' column as target")

    features = df.drop(columns=[c for c in ["Date", "Units_Used_tomorrow"] if c in df.columns])

    # force numeric (non-numeric -> NaN)
    features = features.apply(pd.to_numeric, errors="coerce")

    # drop columns that are fully NaN
    features = features.dropna(axis=1, how="all")

    # ensure target is numeric and handle NaNs
    target = pd.to_numeric(df["Units_Used_tomorrow"], errors="coerce")
    if target.isna().any():
        print(f"Warning: found {int(target.isna().sum())} NaN(s) in 'Units_Used_tomorrow'; imputing.")
        # try forward/back fill to preserve sequence continuity
        target = target.fillna(method="ffill").fillna(method="bfill")
        # if still NaN (all values were NaN), fill with mean (will be NaN then - guard)
        if target.isna().any():
            mean_val = target.mean()
            if pd.isna(mean_val):
                mean_val = 0.0
            target = target.fillna(mean_val)
    target = target.values.astype(float)
    # drop zero-variance columns
    non_constant_cols = features.columns[features.nunique() > 1]
    features = features[non_constant_cols]
    features = features.fillna(features.mean())

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_seq, y_seq = create_sequences(features, target, seq_len)

    split = int(1 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    return X_train, X_test, y_train, y_test


def save_processed(X_train, y_train, out_path):
    # save tensors in a format easy to load and convert to DataLoader in other scripts
    torch.save({"X_train": X_train, "y_train": y_train}, out_path)


def build_dataloader_from_tensors(X, y, batch_size=32, shuffle=True):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class Client:
    def __init__(self, device, train_loader, model, lr=1e-3, server_ip="127.0.0.1", port=8765):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.lr = lr
        self.comms = Client_Com(server_ip, port)
        try:
            self.client_id = self.comms.recieve_id()
        except Exception:
            self.client_id = None

    def train(self, epochs=1):
        # receive global weights
        try:
            global_weights = self.comms.recieve_weights()
            print("recieved weights")
            self.model.load_state_dict(global_weights)
        except Exception as e:
            print(f"Failed to receive weights: {e}")
            return

        self.model.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(epochs):
            print("training epoch, ", epoch)
            total_loss = 0
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Detected NaN/Inf loss — dumping diagnostics:")
                    try:
                        print(f"X any NaN: {torch.isnan(X).any().item()}, any Inf: {torch.isinf(X).any().item()}")
                        print(f"y any NaN: {torch.isnan(y).any().item()}, any Inf: {torch.isinf(y).any().item()}")
                        print(f"X min/max: {X.min().item() if X.numel() else 'NA'}/{X.max().item() if X.numel() else 'NA'}")
                        print(f"y min/max: {y.min().item()}/{y.max().item()}")
                        print(f"outputs min/max: {outputs.min().item()}/{outputs.max().item()}")
                    except Exception as e:
                        print(f"Diagnostics failed: {e}")
                    return
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(self.train_loader))
            print(f"Client {self.client_id} | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # send updated weights
        try:
            self.comms.send(self.model.state_dict())
            print("sending weights")
        except Exception as e:
            print(f"Failed to send weights: {e}")

    def start(self):
        print("starting client loop")
        while True:
            try:
                self.train()
            except KeyboardInterrupt:
                print("client shutting down")
                break
            except Exception as e:
                print(f"Client {self.client_id} error: {e}")
                break


def main():
    parser = argparse.ArgumentParser(description="Process blood bank CSV into client-ready tensors")
    parser.add_argument("index", nargs='?', default=None, help="index number used in CSV filename (e.g. 2 -> blood_bank_data_2.csv)")
    parser.add_argument("--seq", type=int, default=7, help="sequence length")
    parser.add_argument("--out", default=None, help="output .pt path (defaults to processed file next to CSV)")
    parser.add_argument("--batch-size", type=int, default=32, help="local training batch size")
    parser.add_argument("--epochs", type=int, default=1, help="local training epochs per round")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--server-ip", default="127.0.0.1", help="server IP for Client_Com")
    parser.add_argument("--port", type=int, default=8765, help="server port for Client_Com")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="device to run training on")
    args = parser.parse_args()


    idx = args.index
    if idx is None:
        print("Please provide an index number for the CSV filename (e.g. 2 -> blood_bank_data_2.csv)")
        return
    csv_name = f"blood_bank_data_{idx}.csv"

    try:
        X_train, X_test, y_train, y_test = load_and_process(csv_name, seq_len=args.seq)
    except FileNotFoundError:
        print(f"CSV file not found: {csv_name}")
        return
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return

    out_path = args.out or f"blood_bank_data_{idx}_processed.pt"
    save_processed(X_train, y_train, out_path)

    print(f"Processed CSV '{csv_name}' -> saved tensors to '{out_path}'")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(X_train.shape[2])
   
    device = torch.device(args.device)
    loader = build_dataloader_from_tensors(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    print(X_train.shape[2])
    model = Model(X_train.shape[2])
    client = Client(device=device, train_loader=loader, model=model, lr=args.lr, server_ip=args.server_ip, port=args.port)
    # run one training round then keep listening in start()
    try:
        client.start()
    except Exception as e:
        print(f"Client failed to start: {e}")


if __name__ == "__main__":
    main()
