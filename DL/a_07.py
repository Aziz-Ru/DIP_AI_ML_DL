import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import cv2

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

IMG_SIZE = 32

MODEL_PATH = 'simple_cnn.pth'


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


# =========================================================
# DATA
# =========================================================

def get_dataloader(batch_size=128):

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=train_transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
    )

    return trainloader, testloader


# =========================================================
# MODEL
# =========================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.feature = nn.Sequential(

            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.feature(x)

        x = self.gap(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


# =========================================================
# TRAIN
# =========================================================

def train_model(
    epochs=10,
    lr=0.001,
    save_path=MODEL_PATH
):

    trainloader, _ = get_dataloader()

    model = SimpleCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):

        model.train()

        pbar = tqdm.tqdm(trainloader)

        running_loss = 0

        for img, label in pbar:

            img = img.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()

            output = model(img)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            pbar.set_description(
                f"Epoch {epoch+1} Loss {loss.item():.4f}"
            )

        print(f'Epoch {epoch+1} Complete')

    torch.save(model.state_dict(), save_path)

    print(f"Model saved to {save_path}")


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model_path=MODEL_PATH):

    model = SimpleCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.eval()

    return model


# =========================================================
# IMAGE LOADER
# =========================================================

def load_image(img_path):

    img = Image.open(img_path).convert('RGB')

    img_tensor = test_transform(img).to(DEVICE)

    return img, img_tensor


# =========================================================
# PREDICT
# =========================================================

def predict(model, img_tensor):

    with torch.no_grad():

        output = model(img_tensor.unsqueeze(0))

        pred_class = output.argmax(dim=1).item()

    return pred_class


# =========================================================
# NORMALIZE MAP
# =========================================================

def normalize_map(x):

    x = x - np.min(x)

    x = x / (np.max(x) + 1e-8)

    return x


# =========================================================
# CAM
# =========================================================

def generate_cam(model, img_tensor, target_class):

    features = model.feature(
        img_tensor.unsqueeze(0)
    )

    weights = model.classifier.weight[target_class]

    cam = torch.zeros(
        features.shape[2:],
        device=DEVICE
    )

    for k in range(weights.shape[0]):

        cam += weights[k] * features[0, k]

    cam = torch.relu(cam)

    cam = cam.detach().cpu().numpy()

    cam = normalize_map(cam)

    return cam


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(
    model,
    img_tensor,
    target_class,
    target_layer=None
):

    gradients = []
    activations = []

    if target_layer is None:
        target_layer = model.feature[6]

    def forward_hook(module, input, output):

        activations.append(output)

    def backward_hook(module, grad_input, grad_output):

        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    output = model(img_tensor.unsqueeze(0))

    score = output[0, target_class]

    model.zero_grad()

    score.backward()

    grads = gradients[0]

    acts = activations[0]

    weights = grads.mean(
        dim=(2, 3),
        keepdim=True
    )

    gradcam = (weights * acts).sum(dim=1)

    gradcam = torch.relu(gradcam)

    gradcam = gradcam.squeeze()

    gradcam = gradcam.detach().cpu().numpy()

    gradcam = normalize_map(gradcam)

    forward_handle.remove()
    backward_handle.remove()

    return gradcam


# =========================================================
# INTEGRATED GRADIENTS
# =========================================================

def integrated_gradients(
    model,
    img_tensor,
    target_class,
    baseline=None,
    steps=50
):

    if baseline is None:

        baseline = torch.zeros_like(img_tensor)

    scaled_inputs = []

    for i in range(steps + 1):

        alpha = i / steps

        scaled = baseline + alpha * (
            img_tensor - baseline
        )

        scaled_inputs.append(scaled)

    grads = []

    for scaled in scaled_inputs:

        scaled = scaled.unsqueeze(0)

        scaled.requires_grad = True

        output = model(scaled)

        score = output[0, target_class]

        model.zero_grad()

        score.backward()

        grad = scaled.grad.detach().cpu()

        grads.append(grad)

    grads = torch.stack(grads)

    avg_grads = grads.mean(dim=0)

    ig = (
        img_tensor.cpu() - baseline.cpu()
    ) * avg_grads.squeeze(0)

    ig = ig.abs().sum(dim=0)

    ig = ig.numpy()

    ig = normalize_map(ig)

    return ig


# =========================================================
# VISUALIZATION
# =========================================================

def overlay_heatmap(img_np, cam_map):

    cam_map = cv2.resize(
        cam_map,
        (IMG_SIZE, IMG_SIZE)
    )

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_map),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    heatmap = np.float32(heatmap) / 255

    overlay = heatmap + img_np

    overlay = overlay / np.max(overlay)

    return heatmap, overlay


# =========================================================
# PLOT
# =========================================================

def plot_results(
    img_np,
    attribution_map,
    overlay,
    title='CAM',
    save_path=None
):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(attribution_map, cmap='jet')
    plt.title(title)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')

    if save_path is not None:

        plt.savefig(save_path)

    plt.show()


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def explain_image(
    img_path,
    method='cam',
    save_path=None
):

    model = load_model()

    _, img_tensor = load_image(img_path)

    pred_class = predict(model, img_tensor)

    print(
        f"Predicted: {CLASSES[pred_class]}"
    )

    img_np = img_tensor.permute(
        1,
        2,
        0
    ).cpu().numpy()

    img_np = normalize_map(img_np)

    # -----------------------------------------------------

    if method == 'cam':

        attr_map = generate_cam(
            model,
            img_tensor,
            pred_class
        )

        title = 'CAM'

    elif method == 'gradcam':

        attr_map = generate_gradcam(
            model,
            img_tensor,
            pred_class
        )

        title = 'Grad-CAM'

    elif method == 'ig':

        attr_map = integrated_gradients(
            model,
            img_tensor,
            pred_class
        )

        title = 'Integrated Gradients'

    else:

        raise ValueError(
            'method must be cam/gradcam/ig'
        )

    heatmap, overlay = overlay_heatmap(
        img_np,
        attr_map
    )

    plot_results(
        img_np,
        attr_map,
        overlay,
        title=title,
        save_path=save_path
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':

    # train_model()

    explain_image(
        './img0001.jpg',
        method='cam',
        save_path='cam_result.png'
    )

    explain_image(
        './img0001.jpg',
        method='gradcam',
        save_path='gradcam_result.png'
    )

    explain_image(
        './img0001.jpg',
        method='ig',
        save_path='ig_result.png'
    )