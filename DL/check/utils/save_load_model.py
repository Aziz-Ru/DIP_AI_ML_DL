import torch
from torchvision import models

def load_mobnetv2_model(path: str,classifier_output_size=2):
  model = models.mobilenet_v2(weights=None)
  model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(model.classifier[1].in_features, classifier_output_size)
  )
  model.load_state_dict(torch.load(path))
  
  return model