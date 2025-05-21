import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18


global testset

# Obtención de los datos de test del dataset CIFAR-10, adaptados para usar en ResNet18 
# (si los datos ya están instalados, la función simplemente retorna la variable que los referencia)
def datos_test():

    # Transformaciones estándar para adaptar CIFAR-10 a ResNet
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    # Descarga del conjunto de pruebas
    testset = torchvision.datasets.CIFAR10(
        root="./CIFAR-10-Python/data", 
        train=False, download=True, 
        transform=transform
    )

    return testset



device = torch.device("cuda")

# Modelo preentrenado
model = resnet18(device=device) 


print("CUDA disponible:", torch.cuda.is_available())
print("CUDA:",torch.version.cuda)
print("GPU detectadas:", torch.cuda.device_count())
print("Nombre del dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Ninguna")





# A partir d'aquí no sé què fa ni si fa falta de moment
# -----------------------------------------------------------

testloader = torch.utils.data.DataLoader(datos_test(), batch_size=32, shuffle=False)


# Ajuste del modelo (opcional): reemplazar la última capa para 10 clases
model.fc = torch.nn.Linear(model.fc.in_features, 10)


