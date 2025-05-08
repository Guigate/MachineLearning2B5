import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------
# Prétraitement des données
# ----------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=False
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# ----------------------
# Chargement des données
# ----------------------

batch_size = 32
img_size = (224, 224)

train_generator = train_datagen.flow_from_directory(
    "dataset/train/",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    "dataset/val/",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_datagen.flow_from_directory(
    "dataset/test/",
    target_size=img_size,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# ----------------------
# Création du modèle basé sur MobileNetV2
# ----------------------

# Charger le modèle MobileNetV2 sans la couche de sortie
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Figer les couches de base pour ne pas les entraîner
base_model.trainable = False

# Ajouter nos couches personnalisées
x = base_model.output
x = GlobalAveragePooling2D()(x)      
x = Dropout(0.3)(x)                  
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(5, activation='softmax')(x)  # 5 classes

# Assembler le modèle
model = Model(inputs=base_model.input, outputs=predictions)

# Compilation
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Affichage du résumé
model.summary()

# ----------------------
# Entraînement du modèle
# ----------------------

# Early stopping pour éviter l'overfitting
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # Attendre 5 epochs sans amélioration
        restore_best_weights=True
    )
]

# Entraînement du modèle
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks,
)

# ----------------------
# Sauvegarde du modèle
# ----------------------

model.save("mon_modele_medicaments_mobilenetv2_v1bis.h5")

# ----------------------
# Évaluation du modèle
# ----------------------

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ----------------------
# Visualisation des résultats
# ----------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Evolution de l'Accuracy et de la Loss")
plt.xlabel('Epochs')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.grid(True)
plt.show()