import os
import shutil
import random

# Paramètres
dataset_dir = ''  # à adapter
original_dirs = ['TRAIN', 'VALID', 'TEST']
new_base_dir = os.path.join(dataset_dir, 'reorganized_dataset')
split_ratios = {'TRAIN': 0.7, 'VALID': 0.15, 'TEST': 0.15}

# Créer les nouveaux dossiers
for split in split_ratios:
    os.makedirs(os.path.join(new_base_dir, split), exist_ok=True)

# Lister les médicaments à partir d'un des dossiers d'origine
sample_dir = os.path.join(dataset_dir, original_dirs[0])
med_names = [name for name in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, name))]

# Pour chaque médicament
for med in med_names:
    print(f"Traitement de {med}...")

    # Rassembler toutes les images de tous les dossiers (TRAIN, VALID, TEST)
    all_images = []
    for folder in original_dirs:
        med_folder = os.path.join(dataset_dir, folder, med)
        if os.path.exists(med_folder):
            all_images += [os.path.join(med_folder, img) for img in os.listdir(med_folder) if os.path.isfile(os.path.join(med_folder, img))]

    # Mélanger les images
    random.shuffle(all_images)

    # Calcul des tailles
    total = len(all_images)
    TRAIN_end = int(split_ratios['TRAIN'] * total)
    VALID_end = TRAIN_end + int(split_ratios['VALID'] * total)

    # Répartition
    split_data = {
        'TRAIN': all_images[:TRAIN_end],
        'VALID': all_images[TRAIN_end:VALID_end],
        'TEST': all_images[VALID_end:]
    }

    # Copier les images dans la nouvelle structure
    for split, files in split_data.items():
        dest_folder = os.path.join(new_base_dir, split, med)
        os.makedirs(dest_folder, exist_ok=True)
        for filepath in files:
            filename = os.path.basename(filepath)
            shutil.copy(filepath, os.path.join(dest_folder, filename))
