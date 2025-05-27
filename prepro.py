import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import os

BATCH_SIZE = 128
SAVE_DIR = "data"

transformaciones = {
    "normal": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "horizontal_flip": transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "random_rotation": transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "gaussian_blur": transforms.Compose([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "color_jitter": transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

os.makedirs(SAVE_DIR, exist_ok=True)

for name, transform in transformaciones.items():
    print(f"Procesando transformación: {name}")

    dataset = datasets.CIFAR10(root='./CIFAR-10', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_imgs, all_labels = [], []

    for imgs, labels in tqdm(loader):
        all_imgs.append(imgs)
        all_labels.append(labels)

    imgs_tensor = torch.cat(all_imgs)
    labels_tensor = torch.cat(all_labels)

    torch.save({'data': imgs_tensor, 'labels': labels_tensor}, f"{SAVE_DIR}/cifar10_test_{name}.pt")
