import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Model  

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Model()
    model.load_state_dict(torch.load("weights.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X)
            pred = output.argmax(dim=1)
            if pred!=y:
                print(pred.item(),y.item())
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Test accuracy: {acc*100:.2f}%")
