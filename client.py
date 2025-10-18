# client.py
import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, client_id, model, train_loader, device='cpu', lr=0.001):
        """
        Initialize a federated client.

        Args:
            client_id (int): Unique ID of the client.
            model (nn.Module): The model architecture (untrained copy).
            train_loader (DataLoader): Local data loader for this client.
            device (str): 'cpu' or 'cuda'.
            lr (float): Learning rate.
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.lr = lr

    def train(self, global_weights, epochs=1):
        """
        Train the client model using the provided global weights.

        Args:
            global_weights (dict): State dictionary from the server's global model.
            epochs (int): Number of local training epochs.

        Returns:
            dict: Updated model state_dict after local training.
        """
        # Load global weights into local model
        self.model.load_state_dict(global_weights)
        self.model.train()

        # Define optimizer and loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        # Local training
        for epoch in range(epochs):
            total_loss = 0
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Client {self.client_id} | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # Return updated model weights
        return self.model.state_dict()
