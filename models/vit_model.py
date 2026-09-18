"""
Architecture Vision Transformer pour la classification du cancer du sein
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class PatchEmbedding(layers.Layer):
    """Convertit l'image en patches et les projette dans un espace d'embedding"""

    def __init__(self, image_size, patch_size, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Projection des patches
        self.projection = layers.Conv2D(
            filters=projection_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="patch_projection"
        )

    def call(self, images):
        # Shape: (batch, height, width, channels) -> (batch, num_patches, projection_dim)
        patches = self.projection(images)
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(patches, [batch_size, -1, self.projection_dim])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "projection_dim": self.projection_dim,
        })
        return config


class PositionEmbedding(layers.Layer):
    """Ajoute des embeddings de position apprenables"""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        # Token de classification [CLS]
        self.cls_token = self.add_weight(
            shape=[1, 1, projection_dim],
            initializer="random_normal",
            trainable=True,
            name="cls_token"
        )

        # Embeddings de position pour patches + CLS token
        self.position_embedding = self.add_weight(
            shape=[1, num_patches + 1, projection_dim],
            initializer="random_normal",
            trainable=True,
            name="position_embedding"
        )

    def call(self, patches):
        batch_size = tf.shape(patches)[0]

        # Répéter le CLS token pour chaque élément du batch
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])

        # Concaténer CLS token et patches
        patches = tf.concat([cls_tokens, patches], axis=1)

        # Ajouter les embeddings de position
        patches = patches + self.position_embedding

        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config


class TransformerBlock(layers.Layer):
    """Bloc Transformer avec Multi-Head Attention et MLP"""

    def __init__(self, projection_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        # Layer Normalization
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        # Multi-Head Attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=dropout_rate
        )

        # MLP
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout_rate),
            layers.Dense(projection_dim),
            layers.Dropout(dropout_rate)
        ])

    def call(self, inputs, training=False):
        # Multi-Head Attention avec connexion résiduelle
        x1 = self.norm1(inputs)
        attention_output = self.attention(x1, x1, training=training)
        x2 = inputs + attention_output

        # MLP avec connexion résiduelle
        x3 = self.norm2(x2)
        mlp_output = self.mlp(x3, training=training)
        output = x2 + mlp_output

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


def create_vit_model(
    image_size=224,
    patch_size=16,
    num_classes=3,
    projection_dim=768,
    num_heads=12,
    transformer_layers=12,
    mlp_head_units=[2048, 1024],
    dropout_rate=0.1
):
    """
    Crée un modèle Vision Transformer

    Args:
        image_size: Taille de l'image d'entrée
        patch_size: Taille des patches
        num_classes: Nombre de classes
        projection_dim: Dimension de projection
        num_heads: Nombre de têtes d'attention
        transformer_layers: Nombre de blocs Transformer
        mlp_head_units: Unités pour la tête de classification MLP
        dropout_rate: Taux de dropout

    Returns:
        Modèle Keras compilé
    """

    num_patches = (image_size // patch_size) ** 2

    # Input
    inputs = layers.Input(shape=(image_size, image_size, 3), name="input_image")

    # Patch Embedding
    patches = PatchEmbedding(image_size, patch_size, projection_dim)(inputs)

    # Position Embedding
    encoded_patches = PositionEmbedding(num_patches, projection_dim)(patches)

    # Dropout après embeddings
    x = layers.Dropout(dropout_rate)(encoded_patches)

    # Transformer Blocks
    for i in range(transformer_layers):
        x = TransformerBlock(
            projection_dim=projection_dim,
            num_heads=num_heads,
            mlp_dim=projection_dim * 4,
            dropout_rate=dropout_rate,
            name=f"transformer_block_{i}"
        )(x)

    # Layer Normalization finale
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Extraire le CLS token (premier token)
    x = x[:, 0, :]

    # Classification Head (MLP)
    for units in mlp_head_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)

    # Couche de sortie
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    # Créer le modèle
    model = keras.Model(inputs=inputs, outputs=outputs, name="ViT_Breast_Cancer")

    return model


def create_vit_with_pretrained_backbone(
    image_size=224,
    num_classes=3,
    dropout_rate=0.1,
    fine_tune_layers=4
):
    """
    Crée un ViT avec backbone pré-entraîné (transfer learning)

    Args:
        image_size: Taille de l'image
        num_classes: Nombre de classes
        dropout_rate: Taux de dropout
        fine_tune_layers: Nombre de couches à débloquer pour le fine-tuning

    Returns:
        Modèle avec backbone pré-entraîné
    """
    try:
        from vit_keras import vit

        # Charger ViT pré-entraîné sur ImageNet
        base_model = vit.vit_b16(
            image_size=image_size,
            pretrained=True,
            include_top=False,
            pretrained_top=False
        )

        # Geler les couches sauf les dernières
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

        # Ajouter une tête de classification personnalisée
        inputs = layers.Input(shape=(image_size, image_size, 3))
        x = base_model(inputs, training=False)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="ViT_Pretrained")

        return model

    except ImportError:
        print("vit-keras n'est pas installé. Utilisation du ViT personnalisé.")
        return create_vit_model(
            image_size=image_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )


if __name__ == "__main__":
    # Test du modèle
    model = create_vit_model()
    model.summary()

    print("\n=== Architecture du modèle ===")
    print(f"Nombre total de paramètres: {model.count_params():,}")
    print(f"Nombre de paramètres entraînables: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
