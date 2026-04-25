import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from dataloader.cifa10_dataloader import cifar10_dataloader
from utils.train_eval import train,evaluate

class CNNModel(nn.Module):
    def __init__(self, activation='relu',img_size=224,num_classes=2):
        super().__init__()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.act    = nn.LeakyReLU()
        
        self.img_size = img_size

        
        self.conv1  = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
        self.conv2  = nn.Conv2d(32,64,3,padding=1)
        self.conv3  = nn.Conv2d(64,128,3,padding=1)
        
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(128 * (img_size // 8) * (img_size // 8), 512)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.pool(self.act(self.conv3(x)))
        # x = x.view(-1, 128 * (self.img_size // 8) * (self.img_size // 8))
        x = x.view(x.size(0), -1)  # ✅ FIX
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

def get_features(model,img):

    model.eval()
    with torch.no_grad():
        conv1_out = torch.relu(model.conv1(img))
        poll1_out = model.pool(conv1_out)

        conv2_out = torch.relu(model.conv2(poll1_out))
        poll2_out = model.pool(conv2_out)

        conv3_out = torch.relu(model.conv3(poll2_out))
        # poll3_out = model.pool(conv3_out)
    return conv1_out, conv2_out, conv3_out

def show_maps(feature_map, title, max_maps=8,file_name=None):
    # Select FIRST image in batch
    fmap = feature_map[0]   # shape: [C, H, W]

    fmap = fmap.detach().cpu()

    num_maps = min(max_maps, fmap.shape[0])

    plt.figure(figsize=(12, 6))
    for i in range(num_maps):
        plt.subplot(2, 4, i+1)
        plt.imshow(fmap[i], cmap='gray')  # now shape [H, W]
        plt.axis('off')

    plt.suptitle(title)
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()

def main():
    cwd = os.getcwd()
    data_loader_path = os.path.join(cwd, 'dataset', 'person')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = cifar10_dataloader()

    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels shape: {labels.shape}")
    
    model = CNNModel(img_size=32,num_classes=10,activation='leakyrelu').to(device)

    num_epochs = 20
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):

        test_loss,teat_acc=train(model=model,dataloader=train_loader,loss_fn=loss_fn,optimizer=optimizer,device=device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {test_loss:.4f}, Train Acc: {teat_acc:.4f}")
        valid_loss, valid_acc = evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
    
    # Problem --1
    # conv1_out, conv2_out, conv3_out = get_features(model, images)
    # print(f"Conv1 output shape: {conv1_out.shape}")
    # print(f"Conv2 output shape: {conv2_out.shape}")
    # print(f"Conv3 output shape: {conv3_out.shape}")
    # show_maps(conv1_out, "Conv1 Features",file_name=os.path.join(cwd, 'results', 'a_03_conv1_features.png'))
    # show_maps(conv2_out, "Conv2 Features",file_name=os.path.join(cwd, 'results', 'a_03_conv2_features.png'))
    # show_maps(conv3_out, "Conv3 Features",file_name=os.path.join(cwd, 'results', 'a_03_conv3_features.png'))


main()