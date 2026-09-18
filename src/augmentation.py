"""
Augmentation de données pour images mammographiques
"""
import numpy as np
import tensorflow as tf
import albumentations as A
import cv2
from typing import Tuple, Optional


def create_augmentation_pipeline(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_strength: str = 'medium'
) -> A.Compose:
    """
    Crée un pipeline d'augmentation avec Albumentations

    Args:
        image_size: Taille de l'image
        augmentation_strength: 'light', 'medium', ou 'strong'

    Returns:
        Pipeline Albumentations
    """

    if augmentation_strength == 'light':
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
            ),
        ]

    elif augmentation_strength == 'medium':
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=50, p=0.3
            ),
        ]

    elif augmentation_strength == 'strong':
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=20, p=0.6),
            A.ShiftScaleRotate(
                shift_limit=0.15, scale_limit=0.15, rotate_limit=20, p=0.6
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.6
            ),
            A.GaussNoise(var_limit=(10.0, 80.0), p=0.4),
            A.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=50, p=0.4
            ),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
        ]
    else:
        raise ValueError(f"augmentation_strength doit être 'light', 'medium' ou 'strong'")

    # Ajouter le redimensionnement final
    transforms.append(A.Resize(image_size[0], image_size[1]))

    return A.Compose(transforms)


def augment_image(image: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """
    Applique l'augmentation à une image

    Args:
        image: Image numpy array
        pipeline: Pipeline Albumentations

    Returns:
        Image augmentée
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    augmented = pipeline(image=image)
    augmented_image = augmented['image'].astype(np.float32) / 255.0

    return augmented_image


class MammogramAugmentation:
    """Classe pour gérer l'augmentation spécifique aux mammographies"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        augmentation_strength: str = 'medium',
        preserve_breast_tissue: bool = True
    ):
        """
        Args:
            image_size: Taille des images
            augmentation_strength: Intensité de l'augmentation
            preserve_breast_tissue: Préserver la structure du tissu mammaire
        """
        self.image_size = image_size
        self.augmentation_strength = augmentation_strength
        self.preserve_breast_tissue = preserve_breast_tissue
        self.pipeline = create_augmentation_pipeline(image_size, augmentation_strength)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Applique l'augmentation"""
        return augment_image(image, self.pipeline)

    def augment_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Augmente un batch d'images

        Args:
            images: Batch de shape (batch_size, H, W, C)

        Returns:
            Batch augmenté
        """
        augmented_batch = []
        for img in images:
            aug_img = self(img)
            augmented_batch.append(aug_img)
        return np.array(augmented_batch)


def create_tf_augmentation_layer(image_size: int = 224) -> tf.keras.Sequential:
    """
    Crée une couche d'augmentation TensorFlow intégrée au modèle

    Args:
        image_size: Taille de l'image

    Returns:
        Couche Sequential TensorFlow
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.2),
    ], name="data_augmentation")


def mixup(images: tf.Tensor, labels: tf.Tensor, alpha: float = 0.2) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Applique MixUp pour l'augmentation

    Args:
        images: Batch d'images
        labels: Labels correspondants
        alpha: Paramètre de la distribution Beta

    Returns:
        Images et labels mixés
    """
    batch_size = tf.shape(images)[0]

    # Générer lambda depuis une distribution Beta
    lam = tf.random.uniform([], 0, 1)
    lam = tf.maximum(lam, 1 - lam)

    # Mélanger les indices
    indices = tf.random.shuffle(tf.range(batch_size))

    # Mélanger les images et labels
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


def cutmix(images: tf.Tensor, labels: tf.Tensor, alpha: float = 1.0) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Applique CutMix pour l'augmentation

    Args:
        images: Batch d'images
        labels: Labels correspondants
        alpha: Paramètre de la distribution Beta

    Returns:
        Images et labels avec CutMix
    """
    batch_size = tf.shape(images)[0]
    image_height = tf.shape(images)[1]
    image_width = tf.shape(images)[2]

    # Générer lambda
    lam = tf.random.uniform([], 0, 1)

    # Générer les coordonnées du rectangle
    cut_ratio = tf.math.sqrt(1.0 - lam)
    cut_h = tf.cast(cut_ratio * tf.cast(image_height, tf.float32), tf.int32)
    cut_w = tf.cast(cut_ratio * tf.cast(image_width, tf.float32), tf.int32)

    cx = tf.random.uniform([], 0, image_width, dtype=tf.int32)
    cy = tf.random.uniform([], 0, image_height, dtype=tf.int32)

    x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_width)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_width)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_height)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_height)

    # Mélanger les indices
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)

    # Appliquer CutMix
    mask = tf.zeros((batch_size, image_height, image_width, 1))
    mask = tf.tensor_scatter_nd_update(
        mask,
        tf.stack(tf.meshgrid(tf.range(batch_size), tf.range(y1, y2), tf.range(x1, x2)), axis=-1),
        tf.ones((batch_size, y2 - y1, x2 - x1, 1))
    )

    mixed_images = images * (1 - mask) + shuffled_images * mask

    # Ajuster les labels
    lam = 1 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(image_height * image_width, tf.float32)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


def create_augmented_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    augmentation_factor: int = 3,
    augmentation_strength: str = 'medium',
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée un dataset augmenté à partir des images originales

    Args:
        images: Images originales
        labels: Labels correspondants
        augmentation_factor: Nombre d'images augmentées par image originale
        augmentation_strength: Intensité de l'augmentation
        image_size: Taille des images

    Returns:
        Images et labels augmentés
    """
    augmentor = MammogramAugmentation(
        image_size=image_size,
        augmentation_strength=augmentation_strength
    )

    augmented_images = []
    augmented_labels = []

    # Garder les images originales
    augmented_images.extend(images)
    augmented_labels.extend(labels)

    # Générer les images augmentées
    for img, label in zip(images, labels):
        for _ in range(augmentation_factor):
            aug_img = augmentor(img)
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)


if __name__ == "__main__":
    print("Module d'augmentation chargé avec succès")
    print("\nFonctions disponibles:")
    print("- create_augmentation_pipeline()")
    print("- augment_image()")
    print("- MammogramAugmentation")
    print("- create_tf_augmentation_layer()")
    print("- mixup()")
    print("- cutmix()")
    print("- create_augmented_dataset()")
