import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Utilisation de l'appareil : {device}")

# ----- Paramètres -----
train_dir = "reorganized_dataset/TRAIN"  # Dossier pour les données d'entraînement
val_dir = "reorganized_dataset/VALID"    # Dossier pour les données de validation
batch_size = 32
num_classes = len(os.listdir(train_dir))
img_size = 224

# ----- Transforms -----
# Transformations pour les données d'entraînement (avec augmentations)
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Redimensionner les images
    transforms.RandomRotation(20),           # Rotation aléatoire
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # Zoom aléatoire
    transforms.RandomHorizontalFlip(),       # Flip horizontal
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Jitter de couleur
    transforms.ToTensor(),                   # Conversion en tenseur
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalisation
])

# Transformations pour les données de validation et de test (sans augmentations)
val_test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Redimensionner les images
    transforms.ToTensor(),                   # Conversion en tenseur
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalisation
])

# ----- Data -----
# Chargement des ensembles d'entraînement, de validation et de test
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ----- Charger MobileNetV2 pré-entraîné -----
model = models.mobilenet_v2(pretrained=True)

# Adapter la couche de sortie
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# ----- Entraînement -----
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_accuracies = []  # Liste pour stocker l'accuracy d'entraînement par epoch
val_accuracies = []    # Liste pour stocker l'accuracy de validation par epoch

# Initialisation de la meilleure accuracy
best_val_accuracy = 0.0
best_model_weights = None

for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    print(f"Epoch {epoch+1} - Début de l'entraînement...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calcul de l'accuracy pour ce batch
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:  # Afficher la progression toutes les 10 batches
            print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    epoch_accuracy = 100 * correct / total
    train_accuracies.append(epoch_accuracy)
    print(f"Epoch {epoch+1} terminé - Loss totale : {total_loss:.4f} - Accuracy : {epoch_accuracy:.2f}%")

    # ----- Validation -----
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total
    val_accuracies.append(val_accuracy)
    print(f"Validation Accuracy après Epoch {epoch+1}: {val_accuracy:.2f}%")

    # Sauvegarder les meilleurs poids
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_weights = model.state_dict()  # Sauvegarde des meilleurs poids

# Restaurer les meilleurs poids après l'entraînement
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    print(f"Meilleurs poids restaurés avec une Validation Accuracy de {best_val_accuracy:.2f}%")

# Sauvegarde du modèle entraîné
torch.save(model.state_dict(), "mobilenetGPU5_v3.pth")
print("Modèle sauvegardé")

# ----- Évaluation sur l'ensemble de test -----
test_dir = "reorganized_dataset/TEST"  # Dossier pour les données de test
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
test_loss = 0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Calcul de l'accuracy
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_loss /= len(test_loader)
test_accuracy = 100 * test_correct / test_total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# ----- Visualisation des résultats -----
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.title("Évolution de l'Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

