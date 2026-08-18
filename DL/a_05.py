import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.cifa10_dataloader import cifar10_dataloader
from utils.train_eval import train, evaluate
import matplotlib.pyplot as plt

class UnderfitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*32*32, 10)
        )

    def forward(self, x):
        return self.net(x)

class OverfitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)

class DropoutCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Linear(128*32*32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)
    
def plot_curves(train_loss, val_loss, train_acc, val_acc, title,filename=None):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title(title + " Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.title(title + " Accuracy")
    plt.legend()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def main():
# Define a simple feedforward neural network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OverfitCNN().to(device)
    train_loader, test_loader = cifar10_dataloader()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(10):
       tl,ta = train(model, train_loader, criterion, optimizer, device=device)
       train_losses.append(tl)
       train_accs.append(ta)
       print(f"Epoch {epoch+1}, Train Loss: {tl:.4f}, Train Accuracy: {ta:.2f}%")
       vl,va = evaluate(model, test_loader, criterion, device=device)
       val_losses.append(vl)
       val_accs.append(va)
       print(f"Epoch {epoch+1}, Test Loss: {vl:.4f}, Test Accuracy: {va:.2f}%")

    plot_curves(train_losses, val_losses, train_accs, val_accs, "OverfitCNN", filename="results/overfit_cnn_curves.png")

if __name__ == "__main__":
    main()
