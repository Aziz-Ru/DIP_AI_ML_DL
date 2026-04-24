import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from torchvision import models
from dataloader.flower_dataloader import get_flower_dataloader
from dataloader.male_female_dataloader import get_male_female_dataloader
from dataloader.cifa10_dataloader import cifar10_dataloader
from tqdm import tqdm


def train(model,dataloader,loss_fn,optimizer,device):
  model.train()
  total_loss = 0
  correct = 0
  total = 0
  loop = tqdm(dataloader, desc="Training", leave=False)
  for img, label in loop:
    
    img = img.to(device)
    label = label.to(device)
    # forward pass
    optimizer.zero_grad()
    output = model(img)
    loss = loss_fn(output,label)
    # backward pass
    loss.backward()
    optimizer.step()
    # calculate loss
    total_loss += loss.item()
    _, predicted = torch.max(output, 1)
    correct += (predicted == label).sum().item()
    total += label.size(0)

    loop.set_postfix({
            "loss": total_loss / (total / label.size(0)),
            "acc": correct / total
        })

    avg_loss = total_loss/len(dataloader)
    accuracy = correct/total
    
  return avg_loss, accuracy

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for img, label in loop:
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            loss = loss_fn(output, label)
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
            loop.set_postfix({
                "loss": total_loss / (total / label.size(0)),
                "acc": correct / total
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    root_path = os.getcwd()
    dataset_path = os.path.join(root_path,'data', 'men_women')
    chkpt = os.path.join(root_path,'checkpoints')
    os.makedirs(chkpt, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_loader, valid_loader = get_male_female_dataloader(dataset_path, batch_size=32)
    train_loader, valid_loader = cifar10_dataloader(batch_size=32)

    # for testing only get first image

    # img = next(iter(train_loader))[0][0]
    # print(f"Image shape: {img.shape}")  # Should be [3, 224, 224]
    # print(f"Pixel value range: {img.min().item()} to {img.max().item()}")  # Should be normalized values
    # print(f"Image tensor: {img}")  # Print the actual tensor values to verify normalization
    # print("=== Oxford 102 Flower Dataset Summary ===")
    # print(f"Train Images   : {len(train_loader.dataset):,}")
    # print(f"Valid Images   : {len(valid_loader.dataset):,}")
    # print(f"Test Images    : {len(test_loader.dataset):,}")
    # print(f"Number of Classes : {len(set(label for _, label in train_loader.dataset.samples))}")
    # print("-" * 60)

    # FULL model (backbone + classifier)
    # Loads MobileNetV2 architecture
    # Loads pretrained weights from ImageNet (1000 classes)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Freeze backbone layers (optional, can be fine-tuned later)
    for param in model.parameters():
        param.requires_grad = False
    # print(f"Parameters:{sum(p.numel() for p in model.parameters())}")
    # print("Backbone (feature extractor) layers are frozen. Only classifier will be trained.")
    # # Count total parameters (very common 🔥)
    
    # print("-" * 60)
    # model.classifier is the final part of the network that converts features → predictions
    # print(model.classifier)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_ftrs, 10)  # 10 class for cifar 10
        # 2 classes: male and female

    )

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    best_valid_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, loss_fn, optimizer, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(chkpt, "cifar10_best_model.pth"))
            print(f"New best model saved with validation accuracy: {best_valid_acc:.4f}")


main()