import os
import torch

from torch.utils.data import Dataset, DataLoader,random_split
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


class MaleFemaleDataset(Dataset):

  def __init__(self,root_path):
    super().__init__()
    self.root_path = root_path
    if not os.path.exists(root_path):
      raise ValueError(f"Root path {root_path} does not exist")
    # list of tuples (path_to_file,label)
    
    self.transforms = get_transforms(mode='train')
    self.samples = []
    # label is either "men or "women"
    for class_name in os.listdir(root_path):
      class_path = os.path.join(root_path, class_name)
      if not os.path.isdir(class_path):
        continue
      label =0 if class_name.lower()=='men' else 1
     
      for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.jpg','.jpeg','.png')):
          img_path = os.path.join(class_path, img_file)
          # print(img_path, label)
          self.samples.append((img_path, label))

    num_male = sum(1 for _, lbl in self.samples if lbl == 0)
    num_female = len(self.samples) - num_male
    print(f"✅ Loaded {len(self.samples)} valid images from {root_path}")
    print(f"   Male   : {num_male}")
    print(f"   Female : {num_female}")
    print(f"   Classes: {sorted(set(class_name for class_name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, class_name))))}")

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, index):
    img_path, label = self.samples[index]
    img = Image.open(img_path).convert('RGB')
    img = self.transforms(img)
    return img, label
  

def get_male_female_dataloader(root_path, batch_size=32):
  full_dataset = MaleFemaleDataset(root_path)
  total_size = len(full_dataset)
  train_size = int(0.8 * total_size)
  valid_size = total_size - train_size
  train_dataset,valid_dataset = random_split(full_dataset, [train_size, valid_size])
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

  return train_dataloader, valid_dataloader