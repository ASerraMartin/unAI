import time
import torch
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# === CONFIGURACIÓN ===
PATH_MODELO = "PATH PROPIO/unAI/CIFAR-10-Python/resnet18_cifar10.pt"
BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else RuntimeError("No hay GPU disponible."))
CLASES_CIFAR = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MODO_MONITOREO = False  # Cambiar a False para modo matrices de confusión


# === TRANSFORMACIONES ===
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


# === CARGAR DATALOADER DE TEST CON UNA TRANSFORMACIÓN ===
# def get_test_loader(transform):
#     test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#     return torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False
#     )
def get_preprocessed_loader(nombre_transformacion):
    path = f"data/cifar10_test_{nombre_transformacion}.pt"
    data = torch.load(path)

    tensor_dataset = torch.utils.data.TensorDataset(data['data'], data['labels'])
    return torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )



# === EVALUAR Y MOSTRAR MATRIZ DE CONFUSIÓN ===
def evaluate(modelo, dataloader, titulo):

    if MODO_MONITOREO:
        start = time.time()
    else:
        all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = modelo(inputs)
            _, preds = torch.max(outputs, 1)

            if not MODO_MONITOREO:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    if MODO_MONITOREO:
        end = time.time()
        print(f"Completado en {end - start:.4f} s")
        return end-start
    else:
        print("OK")
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASES_CIFAR)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(titulo)
        plt.show()



# === MAIN ===
if __name__ == "__main__":

    import os
    # cwd = os.getcwd()
    # if not cwd.endswith("CIFAR-10-Python"):
    #     new_cwd = os.path.join(cwd, "CIFAR-10-Python")
    #     os.chdir(new_cwd)
    print(f"Directorio actual: {os.getcwd()}")


    if DEVICE.type == "cuda":
        print(f"\nDispositivo: {DEVICE} - {torch.cuda.get_device_name(DEVICE)}\n")
    else:
        print(f"\nDispositivo: {DEVICE}\n")

    modelo = models.resnet18()
    modelo.fc = torch.nn.Linear(modelo.fc.in_features, 10)
    modelo.load_state_dict(torch.load(PATH_MODELO))
    modelo.eval()
    modelo.to(DEVICE)

    if MODO_MONITOREO:
        gpu_total = 0
        start_total = time.time()

    for i in range(11 if MODO_MONITOREO else 1):


        if MODO_MONITOREO:
            print(f"\n=== ITERACIÓN {i} ===")
            gpu_loop = 0
            start_loop = time.time()

        for name in ["normal","horizontal_flip", "random_rotation", "gaussian_blur", "color_jitter"]:



            if MODO_MONITOREO:
                print(f"\n=== Evaluando: {name} - Iteración {i} ===")
                # test_loader = get_test_loader(transform)
                test_loader = get_preprocessed_loader(name)
                gpu_loop += evaluate(modelo, test_loader, titulo=f"Matriz de confusión - {name}")
            else:
                print(f"\n=== Evaluando: {name} ===")
                # test_loader = get_test_loader(transform)
                test_loader = get_preprocessed_loader(name)
                evaluate(modelo, test_loader, titulo=f"Matriz de confusión - {name}")

        if MODO_MONITOREO:
            end_loop = time.time()
            gpu_total += gpu_loop
            print(f"\n===================================================")
            print(f"Tiempo de GPU: {gpu_loop:.4f} s - Iteración {i}")
            print(f"Tiempo de ejecución: {end_loop - start_loop:.4f} s - Iteración {i}")
            print(f"===================================================\n")

    
    if MODO_MONITOREO:
        end_total = time.time()
        print(f"\n#########################################################")
        print(f"Tiempo total de GPU: {gpu_total:.4f} s")
        print(f"Tiempo total de ejecución: {end_total - start_total:.4f} s")
        print(f"#########################################################\n")
