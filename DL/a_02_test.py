import torch
from torchvision import models,transforms,datasets
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import umap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_test_dataloader(root_path, batch_size=32):
    test_dataset = datasets.ImageFolder(root=root_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"✅ Loaded {len(test_dataset)} test images from {root_path}")
    return test_loader

def load_mobnetv2_model(path: str,classifier_output_size=2):
  model = models.mobilenet_v2(weights=None)
  model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(model.classifier[1].in_features, classifier_output_size)
  )
  model.load_state_dict(torch.load(path))
  model = model.to(device)
  print("Model loaded successfully from")
  return model


def show_predictions(model, test_loader):
    model.eval()
    model.eval()
    show_imgs = []
    show_labels = []
    show_preds = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            show_imgs.append(imgs.cpu())
            show_labels.append(labels.cpu())
            show_preds.append(preds.cpu())
            # for i in range(len(preds)):
            #     print(f"Pred: {preds[i].item()} | Actual: {labels[i].item()} | Confidence: {probs[i][preds[i]].item()*100:.2f}%")

    show_imgs = torch.cat(show_imgs)
    show_labels = torch.cat(show_labels)
    show_preds = torch.cat(show_preds)

    num_images = len(show_imgs)

    # Plot settings
    cols = 5   # change if you want
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(num_images):
        img = show_imgs[i].permute(1, 2, 0)  # CHW -> HWC

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")

        plt.title(f"P:{show_preds[i].item()} | T:{show_labels[i].item()}")

    plt.tight_layout()
    # plt.show()
    plt.savefig("cifar_train_test_flower.png")

def extract_feature(model,dataloder):
    model.eval()
    feature = []
    labels = []
    with torch.no_grad():
        for imgs,label in dataloder:
            imgs = imgs.to(device)
            label = label.to(device)

            x = model.features(imgs)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)  # Flatten to [batch_size, feature_dim]
            feature.append(x.cpu())
            labels.append(label.cpu())
    feature = torch.cat(feature).numpy()
    labels = torch.cat(labels).numpy()
    return feature, labels


def run_pca(features,labels, file_name='pca'):
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title("PCA of MobileNetV2 Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(f'{file_name}.png')
    # plt.show()

def run_tsne(features,labels,file_name='tsne'):
    n_samples = len(features)
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, random_state=42,perplexity=perplexity)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title("t-SNE of MobileNetV2 Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(f'{file_name}.png')


def run_umap(features,labels,file_name='umap'):
    reducer = umap.UMAP(n_components=2, random_state=42)
    features_2d = reducer.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title("UMAP of MobileNetV2 Features")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.savefig(f'{file_name}.png')

def test_model():
    root_path = os.getcwd()
    flower_path = os.path.join(root_path,'dataset','flower')
    person_path = os.path.join(root_path,'dataset','person')
    # here chanage path of choose best for men & women dataset
    model_path = os.path.join(root_path,'checkpoints', "cifar10_best_model.pth")
    best_model_path = os.path.join(root_path,'checkpoints', "best_model.pth")

    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}. Please train the model first.")
        return
    if not os.path.exists(best_model_path):
        print(f"Best model checkpoint not found at {best_model_path}. Please train the model first.")
        return
    model_imgenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # for men woment outputsize must be 2
    cifar10moderl = load_mobnetv2_model(model_path, classifier_output_size=10)
    bestMenWomenModel = load_mobnetv2_model(best_model_path, classifier_output_size=2)

    person_loader = get_test_dataloader(person_path, batch_size=32)
    flower_loader = get_test_dataloader(flower_path, batch_size=32)

    # show_predictions(model, test_loader)
    # model.eval()
    feature ,label = extract_feature(model_imgenet, flower_loader)
    run_pca(feature,label, file_name='pca_features_flower_imgnet')
    run_tsne(feature,label, file_name='tsne_features_flower_imgnet')
    run_umap(feature,label,file_name='umap_feature_flower_imagenet')

    feature,label = extract_feature(model_imgenet, person_loader)
    run_pca(feature,label, file_name='pca_features_person_imgnet')
    run_tsne(feature,label, file_name='tsne_features_person_imgnet')
    run_umap(feature,label,file_name='umap_feature_person_imagenet')


    feature,label = extract_feature(cifar10moderl, flower_loader)
    run_pca(feature,label, file_name='pca_features_flower_cifar10')
    run_tsne(feature,label, file_name='tsne_features_flower_cifar10')
    run_umap(feature,label,file_name='umap_feature_flower_cifar10')

    feature,label = extract_feature(bestMenWomenModel, person_loader)
    run_pca(feature,label, file_name='pca_features_person_fine_tune')
    run_tsne(feature,label, file_name='tsne_features_person_fine_tune')
    run_umap(feature,label,file_name='umap_feature_person_fine_tune')


    
    

if __name__ == "__main__":
    test_model()