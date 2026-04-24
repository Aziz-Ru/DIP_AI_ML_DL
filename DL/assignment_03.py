import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

mobnet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
vgg16 = models.vgg16()
print(vgg16)
