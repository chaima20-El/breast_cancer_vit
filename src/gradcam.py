"""
Grad-CAM pour visualiser les zones d'attention du modèle Vision Transformer
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os


class GradCAM:
    """Implémentation de Grad-CAM pour Vision Transformers"""

    def __init__(
        self,
        model: keras.Model,
        layer_name: Optional[str] = None
    ):
        """
        Args:
            model: Modèle Keras
            layer_name: Nom de la couche à visualiser (auto-détecté si None)
        """
        self.model = model

        # Trouver automatiquement la dernière couche de convolution ou transformer
        if layer_name is None:
            layer_name = self._find_target_layer()

        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()

    def _find_target_layer(self) -> str:
        """Trouve automatiquement la couche cible pour Grad-CAM"""

        # Pour ViT, on cherche la dernière couche de normalisation avant la classification
        for layer in reversed(self.model.layers):
            if 'layer_normalization' in layer.name.lower():
                return layer.name
            elif 'transformer_block' in layer.name.lower():
                return layer.name
            elif 'conv' in layer.name.lower():
                return layer.name

        # Par défaut, prendre l'avant-dernière couche
        return self.model.layers[-2].name

    def _build_grad_model(self) -> keras.Model:
        """Construit le modèle pour calculer les gradients"""

        try:
            target_layer = self.model.get_layer(self.layer_name)
        except ValueError:
            print(f"Couche '{self.layer_name}' non trouvée. Couches disponibles:")
            for layer in self.model.layers:
                print(f"  - {layer.name}")
            raise

        return keras.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output]
        )

    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None,
        use_guided: bool = False
    ) -> np.ndarray:
        """
        Calcule la heatmap Grad-CAM

        Args:
            image: Image d'entrée (H, W, 3)
            class_idx: Indice de la classe cible (None = classe prédite)
            use_guided: Utiliser Guided Grad-CAM

        Returns:
            Heatmap normalisée [0, 1]
        """
        # Préparer l'image
        img_array = np.expand_dims(image, axis=0)

        # Calculer les gradients
        with tf.GradientTape() as tape:
            layer_output, predictions = self.grad_model(img_array)

            if class_idx is None:
                class_idx = tf.argmax(predictions[0]).numpy()

            class_channel = predictions[:, class_idx]

        # Gradients de la classe par rapport à la couche cible
        grads = tape.gradient(class_channel, layer_output)

        # Guided Grad-CAM
        if use_guided:
            grads = tf.cast(layer_output > 0, "float32") * tf.cast(grads > 0, "float32") * grads

        # Global Average Pooling des gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        # Pondérer la couche de sortie par les gradients
        layer_output = layer_output[0]

        # Pour les Transformers, on peut avoir une shape (num_patches, dim)
        if len(layer_output.shape) == 2:
            heatmap = layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Reconstruire en 2D (patch_size x patch_size)
            num_patches = int(np.sqrt(len(heatmap)))
            if num_patches ** 2 == len(heatmap):
                heatmap = tf.reshape(heatmap, (num_patches, num_patches))
            else:
                # Ignorer le CLS token
                heatmap = heatmap[1:]
                num_patches = int(np.sqrt(len(heatmap)))
                heatmap = tf.reshape(heatmap, (num_patches, num_patches))
        else:
            # Pour les CNN classiques
            heatmap = layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

        # Normaliser
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        return heatmap

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Superpose la heatmap sur l'image

        Args:
            image: Image originale
            heatmap: Heatmap Grad-CAM
            alpha: Transparence de la heatmap
            colormap: Carte de couleurs OpenCV

        Returns:
            Image avec heatmap superposée
        """
        # Redimensionner la heatmap à la taille de l'image
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Convertir en uint8
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Appliquer la colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # Convertir l'image en uint8 si nécessaire
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = np.uint8(255 * image)
        else:
            image_uint8 = image

        # Assurer que l'image est en RGB
        if len(image_uint8.shape) == 2:
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
        elif image_uint8.shape[2] == 1:
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

        # Superposer
        superimposed = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)

        return superimposed

    def visualize(
        self,
        image: np.ndarray,
        class_names: list,
        class_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualise Grad-CAM avec l'image originale et la superposition

        Args:
            image: Image d'entrée
            class_names: Liste des noms de classes
            class_idx: Indice de classe cible
            save_path: Chemin pour sauvegarder la visualisation
        """
        # Calculer la heatmap
        heatmap = self.compute_heatmap(image, class_idx)

        # Prédire la classe
        img_array = np.expand_dims(image, axis=0)
        predictions = self.model.predict(img_array, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]

        if class_idx is None:
            class_idx = predicted_class

        # Créer la superposition
        superimposed = self.overlay_heatmap(image, heatmap)

        # Visualiser
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Image originale
        axes[0].imshow(image)
        axes[0].set_title('Image Originale', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Superposition
        axes[2].imshow(superimposed)
        title = f'Classe Prédite: {class_names[predicted_class]}\n'
        title += f'Confiance: {confidence:.2%}'
        if class_idx != predicted_class:
            title += f'\n(Visualisation pour: {class_names[class_idx]})'
        axes[2].set_title(title, fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualisation Grad-CAM sauvegardée: {save_path}")

        plt.show()

    def generate_multiple_gradcams(
        self,
        images: np.ndarray,
        class_names: list,
        output_dir: str,
        num_images: int = 10
    ):
        """
        Génère des visualisations Grad-CAM pour plusieurs images

        Args:
            images: Batch d'images
            class_names: Liste des noms de classes
            output_dir: Répertoire de sortie
            num_images: Nombre d'images à traiter
        """
        os.makedirs(output_dir, exist_ok=True)

        for i in range(min(num_images, len(images))):
            save_path = os.path.join(output_dir, f'gradcam_{i:03d}.png')
            self.visualize(images[i], class_names, save_path=save_path)
            print(f"Traité: {i+1}/{num_images}")


class AttentionMapVisualizer:
    """Visualise les cartes d'attention des Transformers"""

    def __init__(self, model: keras.Model):
        self.model = model

    def extract_attention_weights(
        self,
        image: np.ndarray,
        layer_name: str = None
    ) -> np.ndarray:
        """
        Extrait les poids d'attention d'une couche Transformer

        Args:
            image: Image d'entrée
            layer_name: Nom de la couche MultiHeadAttention

        Returns:
            Poids d'attention
        """
        # Trouver automatiquement une couche d'attention si non spécifiée
        if layer_name is None:
            for layer in self.model.layers:
                if 'multi_head_attention' in layer.name.lower():
                    layer_name = layer.name
                    break

        if layer_name is None:
            raise ValueError("Aucune couche d'attention trouvée")

        # Créer un modèle pour extraire les poids d'attention
        attention_layer = self.model.get_layer(layer_name)

        # Note: L'extraction des poids d'attention nécessite une modification
        # de l'architecture pour retourner les poids explicitement
        # Ceci est une implémentation simplifiée

        print(f"⚠️ L'extraction des poids d'attention nécessite une architecture modifiée")
        print(f"Couche trouvée: {layer_name}")

        return None

    def visualize_attention_map(
        self,
        image: np.ndarray,
        attention_weights: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualise la carte d'attention

        Args:
            image: Image originale
            attention_weights: Poids d'attention
            save_path: Chemin pour sauvegarder
        """
        # Implémentation à venir
        pass


if __name__ == "__main__":
    print("Module Grad-CAM chargé avec succès")
    print("\nClasses disponibles:")
    print("- GradCAM")
    print("- AttentionMapVisualizer")
