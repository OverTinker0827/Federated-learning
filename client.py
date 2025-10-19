# client.py
import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, client_id, model, train_loader, device='cpu', lr=0.001):

        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.lr = lr

    def train(self, global_weights, epochs=1):

        self.model.load_state_dict(global_weights)
        self.model.train()


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)



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


        return self.model.state_dict()
