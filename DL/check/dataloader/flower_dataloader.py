import os
import torch
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def get_transforms(mode='train'):
  if mode=='train':
    return transforms.Compose([
      transforms.Resize((224,224)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(15),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  else:
    return transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class FlowerDataset(Dataset):

  def __init__(self,root_path,mode="train"):
    super().__init__()
    self.root_path = root_path
    self.mode = mode
    self.mode = mode.lower()
    self.transforms = get_transforms(mode)
    if not os.path.exists(root_path):
      raise ValueError(f"Root path {root_path} does not exist")
    
    # list of tuples (path_to_file,label)
    self.samples = []
    self.name_mapping = None
    


    if self.mode in ['train','valid']:
      data_path = os.path.join(root_path,self.mode)
      class_dirs = sorted(os.listdir(data_path))
      
      for dir in class_dirs:
        org_label = int(dir)
        label = org_label-1 # 0 to 101
        class_path = os.path.join(data_path,dir)
        
        for file in os.listdir(class_path):
          if file.lower().endswith(('.jpg','.jpeg','.png')):
            img_path = os.path.join(class_path,file)
            self.samples.append((img_path,label))
    
    elif self.mode=='test':
      data_path = os.path.join(root_path,self.mode)
      
      for file in os.listdir(data_path):
        if file.lower().endswith(('.jpg','.jpeg','.png')):
          img_path = os.path.join(data_path,file)
          self.samples.append((img_path,file))

    else:
      raise ValueError(f"Invalid mode {mode}. Mode should be one of ['train','valid','test']")

    # try:
    #   self.name_mapping =
    # except: 
    #   self.name_mapping = None

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, index):
    
    img,label = self.samples[index]
    img = Image.open(img).convert('RGB')
    img = self.transforms(img)
    return img, label
  

def get_flower_dataloader(root_path, batch_size=32):
  
  train_dataset = FlowerDataset(root_path, mode='train')
  valid_dataset = FlowerDataset(root_path, mode='valid')
  test_dataset = FlowerDataset(root_path, mode='test')

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_dataloader, valid_dataloader, test_dataloader