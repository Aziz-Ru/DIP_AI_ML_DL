import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT:
# Use the SAME value that was used during training
NUM_CLASSES = 480  # <-- CHANGE THIS

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


class TinyViTFaceID(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.backbone = timm.create_model(
            "tiny_vit_5m_224",
            pretrained=False,
            num_classes=0
        )

        self.classifier = nn.Linear(
            320,   # tiny_vit_5m embedding size
            num_classes
        )

    def forward(self, x):

        feat = self.backbone(x)

        logits = self.classifier(feat)

        return logits


# Create model exactly like training
model = TinyViTFaceID(NUM_CLASSES).to(device)

cwd = os.getcwd()

model.load_state_dict(
    torch.load(
        f"{cwd}/tinyvit_faceid.pth",
        map_location=device
    )
)

model.eval()

# Use only backbone for verification
feature_extractor = model.backbone
feature_extractor.eval()


def preprocess(path):

    img = Image.open(path).convert("RGB")

    img = transform(img)

    img = img.unsqueeze(0)

    return img.to(device)


def get_embedding(img_path):

    img = preprocess(img_path)

    with torch.no_grad():

        emb = feature_extractor(img)

        emb = F.normalize(
            emb,
            p=2,
            dim=1
        )

    return emb


def similarity_score(img1_path, img2_path):

    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)

    score = F.cosine_similarity(
        emb1,
        emb2
    ).item()

    return score


def show_similarity(img1_path, img2_path, img3_path, img4_path):

    score1 = similarity_score(
        img1_path,
        img2_path
    )

    score2 = similarity_score(
        img1_path,
        img3_path
    )

    score3 = similarity_score(
        img1_path,
        img4_path
    )

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    img3 = Image.open(img3_path).convert("RGB")
    img4 = Image.open(img4_path).convert("RGB")

    plt.figure(figsize=(12, 5))

    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title("Reference")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title(
        f"Image 2\nSimilarity={score1:.4f}"
    )
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(img3)
    plt.title(
        f"Image 3\nSimilarity={score2:.4f}"
    )
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(img4)
    plt.title(
        f"Image 4\nSimilarity={score3:.4f}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Image1 ↔ Image2 : {score1:.4f}")
    print(f"Image1 ↔ Image3 : {score2:.4f}")
    print(f"Image1 ↔ Image4 : {score3:.4f}")


show_similarity(
    f"{cwd}/dataset/person/men/img0001.jpg",
    f"{cwd}/dataset/person/men/img0046.jpg",
    f"{cwd}/dataset/person/men/img0003.jpg",
    f"{cwd}/dataset/person/men/img0039.jpg"
)