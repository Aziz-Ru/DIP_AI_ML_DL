import os
import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd
from PIL import Image
import urllib.request

# ====================== CONFIG ======================
DATA_DIR = "/content/data/dataset"

BATCH_SIZE = 16
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ====================== 1. DATA LOADER ======================
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

class_names = dataset.classes
print(f"✅ Loaded {len(dataset)} images | {len(class_names)} classes")
print("Classes:", class_names)

if not os.path.exists("imagenet_classes.txt"):
    print("Downloading ImageNet labels...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt"
    )
with open("imagenet_classes.txt") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

def tokenize(name: str) -> set[str]:
    return set(name.lower().replace("_", " ").replace("-", " ").split())

class_tokens = {cn: tokenize(cn) for cn in class_names}

def is_hit(true_class: str, pred_name: str) -> bool:
    pred_tokens = tokenize(pred_name)
    true_toks   = class_tokens[true_class]
    return bool(true_toks & pred_tokens)


# ====================== 2. PRE-TRAINED MODELS ======================
model_names = [
    "alexnet", "vgg16", "vgg19", "resnet50", "resnet101",
    "densenet121", "mobilenet_v2", "efficientnet_b0",
    "efficientnet_b4", "convnext_tiny"
]

models_dict = {}
for name in model_names:
    try:
        model = getattr(models, name)(weights="DEFAULT")
        model = model.to(DEVICE)
        model.eval()
        models_dict[name] = model
        print(f"✅ Loaded {name}")
    except Exception as e:
        print(f"⚠️  Could not load {name}: {e}")


# ====================== 3. CLASSIFICATION (Top-1 & Top-5) ======================
print("\n" + "="*80)
print("RUNNING CLASSIFICATION (Top-1 & Top-5)")
print("="*80)

for model_name, model in models_dict.items():
    print(f"\n🔥 Model: {model_name}")
    all_preds    = []
    correct_top1 = 0
    top5_hits    = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs   = F.softmax(outputs, dim=1)

            _, top5_idx = torch.topk(probs, 5, dim=1)
            top1_idx    = top5_idx[:, 0]

            for i in range(images.size(0)):
                true_class = class_names[labels[i].item()]
                top1_name  = imagenet_classes[top1_idx[i].item()]
                top5_names = [imagenet_classes[idx.item()] for idx in top5_idx[i]]

                all_preds.append({
                    "model":      model_name,
                    "true_class": true_class,
                    "top1":       top1_name,
                    "top5":       " | ".join(top5_names),
                })

                if is_hit(true_class, top1_name):
                    correct_top1 += 1
                if any(is_hit(true_class, n) for n in top5_names):
                    top5_hits += 1

    print(f"   Top-1 loose accuracy : {correct_top1 / len(dataset) * 100:.2f}%")
    print(f"   Top-5 hit rate       : {top5_hits   / len(dataset) * 100:.2f}%")

    df = pd.DataFrame(all_preds)
    df.to_csv(f"results_{model_name}.csv", index=False)
    print(f"   → Saved results_{model_name}.csv")


# ====================== 4. ONE IMAGE PER CLASS — 3×3 GRID ======================
print("\n" + "="*80)
print("SAMPLE PREDICTIONS — 1 image per class (3 cols × 3 rows)")
print("="*80)

# ── pick the first image found for each class ─────────────────────────────────
one_per_class: dict[str, tuple[str, int]] = {}

for img_path, label_idx in dataset.samples:
    cn = class_names[label_idx]
    if cn not in one_per_class:
        one_per_class[cn] = (img_path, label_idx)
    if len(one_per_class) == len(class_names):
        break

display_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
])
infer_transform = transform

# ── fixed 3-column, 3-row grid ───────────────────────────────────────────────
N_COLS    = 3
N_ROWS    = 3                          # 3×3 = 9 cells; 7 classes → 2 empty cells
n_models  = len(models_dict)

# per-cell height: image (3.5 in) + text rows (header × 3 + model × n_models)
cell_h = 3.5 + 0.22 * (n_models + 3)
cell_w = 3.8

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(cell_w * N_COLS, cell_h * N_ROWS))

for idx in range(N_ROWS * N_COLS):
    row = idx // N_COLS
    col = idx  % N_COLS
    ax  = axes[row][col]

    if idx >= len(class_names):
        # blank cell for the 8th and 9th slots
        ax.axis("off")
        continue

    cn = class_names[idx]
    img_path, label_idx = one_per_class[cn]
    filename = os.path.basename(img_path)

    # ── show image ────────────────────────────────────────────────
    pil_img     = Image.open(img_path).convert("RGB")
    display_img = display_transform(pil_img)
    ax.imshow(display_img)
    ax.axis("off")

    # ── run all models ────────────────────────────────────────────
    tensor_img = infer_transform(pil_img).unsqueeze(0).to(DEVICE)
    pred_lines = []
    with torch.no_grad():
        for mname, mmodel in models_dict.items():
            out      = mmodel(tensor_img)
            prob     = F.softmax(out, dim=1)
            top1_idx = prob.argmax(dim=1).item()
            top1_lbl = imagenet_classes[top1_idx]
            short    = (top1_lbl[:22] + "…") if len(top1_lbl) > 22 else top1_lbl
            tick     = "✓" if is_hit(cn, top1_lbl) else "✗"
            pred_lines.append(f"{tick} {mname[:12]}: {short}")

    title_lines = [
        f"True: {cn}",
        f"File: {filename}",
        "─" * 28,
    ] + pred_lines

    ax.set_title(
        "\n".join(title_lines),
        fontsize=6.5,
        loc="left",
        pad=4,
        fontfamily="monospace",
    )

fig.suptitle(
    "One Image per Class — Predictions from All Models\n"
    "(✓ correct  ✗ mismatch)",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

plt.tight_layout()
out_fname = "predictions_one_per_class.png"
plt.savefig(out_fname, dpi=180, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out_fname}")


# ====================== 5. FEATURE EXTRACTION HELPERS ======================

def get_feature_extractor(model_name: str, model: nn.Module):
    name = model_name.lower()
    if "resnet" in name or "convnext" in name:
        return model.avgpool, True
    if "vgg" in name:
        return model.features[-1], True
    if "densenet" in name:
        return model.features, True
    if "efficientnet" in name:
        return model.avgpool, True
    if "mobilenet" in name:
        return model.features[-1], True
    if "alexnet" in name:
        return model.classifier[5], False
    children = list(model.children())
    return children[-2], True


# ====================== 6. FEATURE EXTRACTION + VISUALIZATION ======================
print("\n" + "="*80)
print("FEATURE EXTRACTION + 2D PLOTS (PCA, t-SNE, UMAP)")
print("="*80)

selected_models = ["resnet50", "vgg16", "efficientnet_b0", "convnext_tiny"]

for model_name in selected_models:
    if model_name not in models_dict:
        print(f"⚠️  {model_name} not loaded, skipping.")
        continue

    model = models_dict[model_name]
    print(f"\n📊 Processing {model_name}...")

    features_list = []
    labels_list   = []

    hook_layer, _ = get_feature_extractor(model_name, model)

    def make_hook(feat_list):
        def hook(module, inp, out):
            feat = out.detach().cpu().float()
            if feat.dim() == 4:
                feat = feat.mean(dim=[2, 3])
            elif feat.dim() == 3:
                feat = feat.mean(dim=1)
            feat_list.append(feat.numpy())
        return hook

    hook = hook_layer.register_forward_hook(make_hook(features_list))

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            _      = model(images)
            labels_list.extend(labels.cpu().numpy())

    hook.remove()

    features = np.concatenate(features_list, axis=0)
    labels   = np.array(labels_list)

    print(f"   Extracted features shape: {features.shape}")
    assert features.ndim == 2, f"Expected 2D, got {features.ndim}D"

    n_samples, n_feats = features.shape
    if n_feats > 50 and n_samples > 50:
        n_comp           = min(50, n_samples, n_feats)
        features_reduced = PCA(n_components=n_comp, random_state=42).fit_transform(features)
    else:
        features_reduced = features

    techniques = {
        "PCA":   PCA(n_components=2, random_state=42),
        "t-SNE": TSNE(
            n_components=2,
            perplexity=min(30, max(5, n_samples // 4)),
            random_state=42,
        ),
        "UMAP":  umap.UMAP(n_components=2, random_state=42),
    }

    for tech_name, reducer in techniques.items():
        src     = features if tech_name == "PCA" else features_reduced
        reduced = reducer.fit_transform(src)

        plt.figure(figsize=(12, 9))
        sns.scatterplot(
            x=reduced[:, 0], y=reduced[:, 1],
            hue=[class_names[l] for l in labels],
            palette="tab20",
            s=120,
            alpha=0.85,
        )
        plt.title(f"{model_name.upper()} — {tech_name} Feature Visualization")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fname = f"viz_{model_name}_{tech_name.lower().replace('-', '')}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   Saved: {fname}")

print("\n🎉 All done!")
print("Check the .csv files and .png plots in your folder for the report.")
