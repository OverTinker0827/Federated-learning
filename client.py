# client.py
import torch
import torch.nn as nn
import torch.optim as optim
from client_com import Client_Com
import sys
#in second version model will also be shared
class Client:
    def __init__(self,device,train_loader,model, lr=0.001):

        # self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.lr = lr
        self.comms=Client_Com("127.0.0.1",8765)
        self.client_id=self.comms.recieve_id()
        

    def train(self, epochs=1):
        global_weights=self.comms.recieve_weights()
        print("recieved weights")
        self.model.load_state_dict(global_weights)
        self.model.train()


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)



        for epoch in range(epochs):
            print("training epoch, ",epoch)
            total_loss = 0
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                print(X,y)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Client {self.client_id} | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")


        self.comms.send(self.model.state_dict())
        print("sending weights")
    def start(self):
        print("starting")
        while True:
            try:
                self.train()
            except KeyboardInterrupt:
                print("client shutting down")
                break
            except Exception as e:
                print(f"Client {self.client_id} error: {e}")
                break





