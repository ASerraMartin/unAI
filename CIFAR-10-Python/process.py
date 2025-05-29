import torch
from torchvision import datasets, transforms
import os

# Transforms
test_transforms = {
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
    ]),
}

def get_test_loader(transform):
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

def main():
    os.makedirs("preprocessed", exist_ok=True)

    for name, transform in test_transforms.items():
        loader = get_test_loader(transform)
        images, labels = next(iter(loader))

        torch.save({"images": images, "labels": labels}, "preprocessed/{}.pt".format(name))

if __name__ == "__main__":
    main()