import time
import torch
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# === CONFIGURACIÓN ===
PATH_MODELO = "C:/Users/Usuario/Desktop/unAI/CIFAR-10-Python/resnet18_cifar10.pt"   # Escribir el path ABSOLUTO propio
BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else RuntimeError("No hay GPU disponible."))
CLASES_CIFAR = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MODO_MONITOREO = True  # Cambiar a False para modo matrices de confusión


# === CARGAR DATOS DE TEST PREPROCESADOS ===
def get_preprocessed_loader(nombre_transformacion):
    path = f"data/cifar10_test_{nombre_transformacion}.pt"
    data = torch.load(path)

    tensor_dataset = torch.utils.data.TensorDataset(data['data'], data['labels'])
    return torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


# === EVALUAR Y MOSTRAR MATRIZ DE CONFUSIÓN (si no está en modo monitoreo) ===
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
        print(f"{end - start:.4f}")
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

    print(f"\nDispositivo: {DEVICE} - {torch.cuda.get_device_name(DEVICE)}\n")

    modelo = models.resnet18()
    modelo.fc = torch.nn.Linear(modelo.fc.in_features, 10)
    modelo.load_state_dict(torch.load(PATH_MODELO))
    modelo.eval()
    modelo.to(DEVICE)

    if MODO_MONITOREO:
        gpu_total = 0
        start_total = time.time()

    for i in range(100 if MODO_MONITOREO else 1):

        if MODO_MONITOREO:
            print(f"\n=== ITERACIÓN {i+1} ===")
            gpu_loop = 0
            start_loop = time.time()

        for name in ["normal","horizontal_flip", "random_rotation", "gaussian_blur", "color_jitter"]:

            if MODO_MONITOREO:
                test_loader = get_preprocessed_loader(name)
                gpu_loop += evaluate(modelo, test_loader, titulo=f"Matriz de confusión - {name}")
            else:
                test_loader = get_preprocessed_loader(name)
                evaluate(modelo, test_loader, titulo=f"Matriz de confusión - {name}")

        if MODO_MONITOREO:
            end_loop = time.time()
            gpu_total += gpu_loop
            print(f"\n===================================================")
            print(f"GPU: {gpu_loop:.4f}")
            print(f"Exec: {end_loop - start_loop:.4f}")
            print(f"===================================================\n")
    
    if MODO_MONITOREO:
        end_total = time.time()
        print(f"\n#########################################################")
        print(f"Total GPU: {gpu_total:.4f}")
        print(f"Total exec: {end_total - start_total:.4f}")
        print(f"#########################################################\n")
