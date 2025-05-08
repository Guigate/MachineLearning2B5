from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Prédictions sur plusieurs images
# ----------------------

# Charger le modèle sauvegardé
model = load_model("mon_modele_medicaments_mobilenetv2_v0.h5")

# Taille d'entrée attendue par le modèle
img_size = (224, 224)

# Noms des classes (dans l'ordre des dossiers)
class_names = ['Ascozin', 'Bioflu', 'DayZinc', 'Decolgen', 'Myra_E']

# Liste des chemins d'images à prédire
image_paths = [
    "dataset/new_image.jpg",
    "dataset/new_image2.jpg",
    "dataset/new_image3.jpg",
    "dataset/new_image4.jpg",
    "dataset/new_image5.jpg"
]

# Liste des noms d'images pour l'affichage
image_names = ['Bioflu', 'Decolgen', 'Myra_E', 'DayZinc', 'Ascozin']

# Affichage dans une grille (2 lignes, 3 colonnes)
plt.figure(figsize=(12, 6))
for i, img_path in enumerate(image_paths):
    # Chargement et prétraitement de l'image
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prédiction
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction[0])
    predicted_label = class_names[predicted_class]

    # Affichage de l'image avec le label prédit, la vérité et la précision
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Prédiction : {predicted_label} \nVérité : {image_names[i]} \nPrécision : {prediction[0][predicted_class] * 100:.2f}%")
    plt.axis("off")

plt.tight_layout()
plt.show()