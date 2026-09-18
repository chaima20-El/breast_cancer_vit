"""
Chargement et gestion des données mammographiques
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path
import json


class MammogramDataLoader:
    """Classe pour charger et organiser les données mammographiques"""

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        class_names: List[str] = None
    ):
        """
        Args:
            data_dir: Répertoire contenant les données
            image_size: Taille des images
            batch_size: Taille du batch
            class_names: Liste des noms de classes
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.class_names = class_names or ['benign', 'malignant', 'normal']
        self.num_classes = len(self.class_names)

    def load_from_directory(
        self,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42
    ) -> Tuple:
        """
        Charge les données depuis une structure de répertoires

        Structure attendue:
        data_dir/
            benign/
                image1.png
                image2.png
            malignant/
                image1.png
            normal/
                image1.png

        Args:
            train_split: Proportion de données d'entraînement
            val_split: Proportion de données de validation
            test_split: Proportion de données de test
            random_seed: Seed pour reproductibilité

        Returns:
            (x_train, y_train), (x_val, y_val), (x_test, y_test)
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Les splits doivent sommer à 1.0"

        # Collecter tous les chemins d'images et labels
        image_paths = []
        labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"⚠️ Répertoire manquant: {class_dir}")
                continue

            # Extensions d'images supportées
            extensions = ['.png', '.jpg', '.jpeg', '.dcm', '.tif', '.tiff']

            for ext in extensions:
                paths = list(Path(class_dir).glob(f'*{ext}'))
                image_paths.extend([str(p) for p in paths])
                labels.extend([class_idx] * len(paths))

        if len(image_paths) == 0:
            raise ValueError(f"Aucune image trouvée dans {self.data_dir}")

        print(f"✓ {len(image_paths)} images trouvées")
        for i, class_name in enumerate(self.class_names):
            count = labels.count(i)
            print(f"  - {class_name}: {count} images")

        # Convertir en arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)

        # Split train/temp
        x_train_paths, x_temp_paths, y_train, y_temp = train_test_split(
            image_paths, labels,
            test_size=(val_split + test_split),
            stratify=labels,
            random_state=random_seed
        )

        # Split val/test
        val_ratio = val_split / (val_split + test_split)
        x_val_paths, x_test_paths, y_val, y_test = train_test_split(
            x_temp_paths, y_temp,
            test_size=(1 - val_ratio),
            stratify=y_temp,
            random_state=random_seed
        )

        print(f"\n📊 Distribution des données:")
        print(f"  Train: {len(x_train_paths)} images")
        print(f"  Val:   {len(x_val_paths)} images")
        print(f"  Test:  {len(x_test_paths)} images")

        return (x_train_paths, y_train), (x_val_paths, y_val), (x_test_paths, y_test)

    def create_tf_dataset(
        self,
        image_paths: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = True,
        augment: bool = False,
        preprocess_fn=None
    ) -> tf.data.Dataset:
        """
        Crée un tf.data.Dataset

        Args:
            image_paths: Chemins des images
            labels: Labels
            shuffle: Mélanger les données
            augment: Appliquer l'augmentation
            preprocess_fn: Fonction de prétraitement personnalisée

        Returns:
            Dataset TensorFlow
        """
        # Créer le dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # Shuffle
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))

        # Charger et prétraiter les images
        dataset = dataset.map(
            lambda x, y: self._load_and_preprocess(x, y, preprocess_fn),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Augmentation (si demandée)
        if augment:
            dataset = dataset.map(
                self._augment,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Batching et prefetching
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _load_and_preprocess(
        self,
        image_path: tf.Tensor,
        label: tf.Tensor,
        preprocess_fn=None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Charge et prétraite une image

        Args:
            image_path: Chemin de l'image
            label: Label
            preprocess_fn: Fonction de prétraitement

        Returns:
            (image, label)
        """
        # Lire l'image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, self.image_size)

        # Normaliser à [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # Appliquer prétraitement personnalisé si fourni
        if preprocess_fn is not None:
            image = preprocess_fn(image)

        # One-hot encoding du label
        label = tf.one_hot(label, self.num_classes)

        return image, label

    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applique l'augmentation de données

        Args:
            image: Image
            label: Label

        Returns:
            (image_augmentée, label)
        """
        # Flip horizontal
        image = tf.image.random_flip_left_right(image)

        # Rotation
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

        # Contraste
        image = tf.image.random_contrast(image, 0.8, 1.2)

        # Luminosité
        image = tf.image.random_brightness(image, 0.2)

        # Clipper à [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    def save_dataset_info(self, filepath: str, train_data, val_data, test_data):
        """
        Sauvegarde les informations sur le dataset

        Args:
            filepath: Chemin du fichier de sortie
            train_data: Tuple (paths, labels) pour train
            val_data: Tuple (paths, labels) pour val
            test_data: Tuple (paths, labels) pour test
        """
        info = {
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'train_size': len(train_data[0]),
            'val_size': len(val_data[0]),
            'test_size': len(test_data[0]),
            'class_distribution': {
                'train': {self.class_names[i]: int(np.sum(train_data[1] == i))
                          for i in range(self.num_classes)},
                'val': {self.class_names[i]: int(np.sum(val_data[1] == i))
                        for i in range(self.num_classes)},
                'test': {self.class_names[i]: int(np.sum(test_data[1] == i))
                         for i in range(self.num_classes)}
            }
        }

        with open(filepath, 'w') as f:
            json.dump(info, f, indent=4)

        print(f"✓ Informations du dataset sauvegardées: {filepath}")


def create_sample_dataset(output_dir: str, num_samples_per_class: int = 10):
    """
    Crée un dataset d'exemple pour tester le code

    Args:
        output_dir: Répertoire de sortie
        num_samples_per_class: Nombre d'échantillons par classe
    """
    class_names = ['benign', 'malignant', 'normal']

    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(num_samples_per_class):
            # Créer une image aléatoire
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

            # Sauvegarder
            from PIL import Image
            Image.fromarray(img).save(os.path.join(class_dir, f'sample_{i:03d}.png'))

    print(f"✓ Dataset d'exemple créé dans: {output_dir}")
    print(f"  {len(class_names)} classes × {num_samples_per_class} images = {len(class_names) * num_samples_per_class} images")


def load_from_csv(
    csv_path: str,
    image_dir: str,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple:
    """
    Charge les données depuis un fichier CSV

    Format CSV attendu:
    image_filename,label
    image1.png,benign
    image2.png,malignant

    Args:
        csv_path: Chemin du fichier CSV
        image_dir: Répertoire contenant les images
        image_size: Taille des images

    Returns:
        (image_paths, labels, class_names)
    """
    df = pd.read_csv(csv_path)

    if 'image_filename' not in df.columns or 'label' not in df.columns:
        raise ValueError("Le CSV doit contenir 'image_filename' et 'label'")

    # Construire les chemins complets
    image_paths = [os.path.join(image_dir, fname) for fname in df['image_filename']]

    # Encoder les labels
    class_names = sorted(df['label'].unique())
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    labels = [label_to_idx[label] for label in df['label']]

    print(f"✓ {len(image_paths)} images chargées depuis {csv_path}")
    print(f"  Classes: {class_names}")

    return np.array(image_paths), np.array(labels), class_names


if __name__ == "__main__":
    print("Module de chargement de données chargé avec succès")
    print("\nClasses et fonctions disponibles:")
    print("- MammogramDataLoader")
    print("- create_sample_dataset()")
    print("- load_from_csv()")
