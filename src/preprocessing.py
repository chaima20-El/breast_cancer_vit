"""
Prétraitement des images mammographiques
"""
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from typing import Tuple, Optional


def remove_artifacts(image: np.ndarray) -> np.ndarray:
    """
    Supprime les artefacts et le bruit des mammographies

    Args:
        image: Image en niveaux de gris

    Returns:
        Image nettoyée
    """
    # Débruitage avec filtrage bilatéral
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    return denoised


def apply_clahe(image: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Applique CLAHE (Contrast Limited Adaptive Histogram Equalization)
    pour améliorer le contraste

    Args:
        image: Image en niveaux de gris
        clip_limit: Limite de contraste
        tile_grid_size: Taille de la grille

    Returns:
        Image avec contraste amélioré
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced


def remove_pectoral_muscle(image: np.ndarray) -> np.ndarray:
    """
    Supprime le muscle pectoral (pour les vues MLO)

    Args:
        image: Image mammographique

    Returns:
        Image sans muscle pectoral
    """
    # Seuillage adaptatif
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Opérations morphologiques
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Trouver les contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Garder le plus grand contour (généralement le tissu mammaire)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        result = cv2.bitwise_and(image, mask)
        return result

    return image


def crop_background(image: np.ndarray, threshold=10) -> np.ndarray:
    """
    Recadre le fond noir autour de la mammographie

    Args:
        image: Image mammographique
        threshold: Seuil pour identifier le fond

    Returns:
        Image recadrée
    """
    # Trouver les pixels non noirs
    mask = image > threshold
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return image

    # Trouver la boîte englobante
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Recadrer avec une petite marge
    margin = 10
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(image.shape[0], y_max + margin)
    x_max = min(image.shape[1], x_max + margin)

    cropped = image[y_min:y_max, x_min:x_max]
    return cropped


def normalize_image(image: np.ndarray, method='minmax') -> np.ndarray:
    """
    Normalise l'image

    Args:
        image: Image à normaliser
        method: 'minmax', 'zscore', ou 'clahe'

    Returns:
        Image normalisée
    """
    if method == 'minmax':
        # Normalisation Min-Max [0, 1]
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 0:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = image

    elif method == 'zscore':
        # Z-score normalization
        mean, std = image.mean(), image.std()
        if std > 0:
            normalized = (image - mean) / std
            # Clipper entre -3 et 3 puis normaliser à [0, 1]
            normalized = np.clip(normalized, -3, 3)
            normalized = (normalized + 3) / 6
        else:
            normalized = image

    elif method == 'clahe':
        # CLAHE normalization
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        normalized = apply_clahe(image_uint8)
        normalized = normalized.astype(np.float32) / 255.0
    else:
        normalized = image

    return normalized


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    padding_color: int = 0
) -> np.ndarray:
    """
    Redimensionne l'image en conservant le ratio et ajoute du padding

    Args:
        image: Image source
        target_size: (hauteur, largeur) cible
        padding_color: Couleur du padding

    Returns:
        Image redimensionnée avec padding
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculer le ratio de redimensionnement
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Redimensionner
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Créer une image avec padding
    if len(image.shape) == 3:
        padded = np.full((target_h, target_w, image.shape[2]), padding_color, dtype=image.dtype)
    else:
        padded = np.full((target_h, target_w), padding_color, dtype=image.dtype)

    # Centrer l'image redimensionnée
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded


def preprocess_mammogram(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    apply_clahe_enhancement: bool = True,
    remove_pectoral: bool = False,
    normalize_method: str = 'minmax'
) -> np.ndarray:
    """
    Pipeline complet de prétraitement pour une mammographie

    Args:
        image_path: Chemin vers l'image
        target_size: Taille cible (hauteur, largeur)
        apply_clahe_enhancement: Appliquer CLAHE
        remove_pectoral: Supprimer le muscle pectoral
        normalize_method: Méthode de normalisation

    Returns:
        Image prétraitée en RGB
    """
    # Charger l'image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")

    # Supprimer les artefacts
    image = remove_artifacts(image)

    # Recadrer le fond
    image = crop_background(image)

    # Supprimer le muscle pectoral si demandé
    if remove_pectoral:
        image = remove_pectoral_muscle(image)

    # Améliorer le contraste avec CLAHE
    if apply_clahe_enhancement:
        if image.dtype != np.uint8:
            image_uint8 = (normalize_image(image, 'minmax') * 255).astype(np.uint8)
        else:
            image_uint8 = image
        image = apply_clahe(image_uint8)

    # Normaliser
    image = normalize_image(image, method=normalize_method)

    # Redimensionner avec padding
    image = resize_with_padding(image, target_size)

    # Convertir en RGB (3 canaux)
    if len(image.shape) == 2:
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Normaliser à [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    return image


def preprocess_batch(
    image_paths: list,
    target_size: Tuple[int, int] = (224, 224),
    **kwargs
) -> np.ndarray:
    """
    Prétraite un batch d'images

    Args:
        image_paths: Liste des chemins d'images
        target_size: Taille cible
        **kwargs: Arguments pour preprocess_mammogram

    Returns:
        Array numpy de shape (batch_size, height, width, 3)
    """
    batch = []
    for path in image_paths:
        try:
            img = preprocess_mammogram(path, target_size, **kwargs)
            batch.append(img)
        except Exception as e:
            print(f"Erreur lors du prétraitement de {path}: {e}")
            # Ajouter une image noire en cas d'erreur
            batch.append(np.zeros((*target_size, 3), dtype=np.float32))

    return np.array(batch, dtype=np.float32)


if __name__ == "__main__":
    # Test du prétraitement
    import matplotlib.pyplot as plt

    print("Module de prétraitement chargé avec succès")
    print("\nFonctions disponibles:")
    print("- remove_artifacts()")
    print("- apply_clahe()")
    print("- remove_pectoral_muscle()")
    print("- crop_background()")
    print("- normalize_image()")
    print("- resize_with_padding()")
    print("- preprocess_mammogram()")
    print("- preprocess_batch()")
