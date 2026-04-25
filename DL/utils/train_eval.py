from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


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
