import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Paramètres
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 224
num_classes = 5  

# Chargement du modèle
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
try:
    model.load_state_dict(torch.load("mobilenetGPU5_v3.pth"))
except FileNotFoundError:
    print("Error: The file 'mobilenet_v2.pth' was not found. Please ensure it exists in the correct directory.")
    exit(1)
except RuntimeError as e:
    print(f"Error loading the model weights: {e}")
    exit(1)
model = model.to(device)
model.eval()

# Transformation pour les images
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Fonction de prédiction
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    return pred.item()

# Tester toutes les images dans un dossier
def predict_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    print(f"Prédictions pour les images dans le dossier '{folder_path}':")
    A,B,D,E,F=0,0,0,0,0
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            prediction = predict_image(img_path)
            # print(f"Image: {filename} - Prédiction: {prediction}")
            if prediction != 4:
                print(f"Image: {filename} - Prédiction: {prediction}")
            if prediction == 0:
                A+=1
            elif prediction == 1:
                B+=1
            elif prediction == 2:
                D+=1
            elif prediction == 3:
                E+=1
            elif prediction == 4:
                F+=1
    print(f"Nombre de prédictions pour chaque classe :")
    print(f"Classe Ascozin: {A}, {A*100//(A+B+D+E+F)}%")
    print(f"Classe Bioflu: {B}, {B*100//(A+B+D+E+F)}%")
    print(f"Classe Dayzinc: {D}, {D*100//(A+B+D+E+F)}%")
    print(f"Classe Decolgen: {E}, {E*100//(A+B+D+E+F)}%")
    print(f"Classe Myra_E: {F}, {F*100//(A+B+D+E+F)}%")
    print(f"Nombre total d'images: {A+B+D+E+F}")
# Exemple d'utilisation
# predict_folder("test_images")  # Remplacez "test_images" par le chemin de votre dossier

# predict_folder("reorganized_dataset/TEST/DayZinc")  
# predict_folder("reorganized_dataset/TEST/Ascozin")  
# predict_folder("reorganized_dataset/TEST/Bioflu") 
# predict_folder("reorganized_dataset/TEST/Decolgen")  
# predict_folder("reorganized_dataset/TEST/Myra_E")  
# predict_folder("demo")  
predict_folder("TEST/Myra_E")  
