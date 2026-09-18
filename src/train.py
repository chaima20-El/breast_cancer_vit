"""
Script d'entraînement du modèle Vision Transformer
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
import json

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.vit_model import create_vit_model, create_vit_with_pretrained_backbone
from src.data_loader import MammogramDataLoader
from src.metrics import MetricsCalculator, MetricsVisualizer


def setup_gpu():
    """Configure le GPU pour l'entraînement"""
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            if config.GPU_MEMORY_LIMIT:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=config.GPU_MEMORY_LIMIT
                    )]
                )

            print(f"✓ {len(gpus)} GPU(s) détecté(s)")
        except RuntimeError as e:
            print(f"Erreur GPU: {e}")
    else:
        print("⚠️ Aucun GPU détecté, utilisation du CPU")

    # Mixed Precision
    if config.USE_MIXED_PRECISION:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("✓ Mixed precision activée (float16)")


def create_callbacks(model_name: str) -> list:
    """
    Crée les callbacks pour l'entraînement

    Args:
        model_name: Nom du modèle

    Returns:
        Liste de callbacks
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    callbacks = []

    # ModelCheckpoint - sauvegarder le meilleur modèle
    checkpoint_path = os.path.join(model_dir, "best_model.h5")
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)

    # EarlyStopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)

    # ReduceLROnPlateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # TensorBoard
    tensorboard_dir = os.path.join(model_dir, 'tensorboard')
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard)

    # CSV Logger
    csv_path = os.path.join(model_dir, 'training_log.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_path, separator=',', append=False)
    callbacks.append(csv_logger)

    print(f"✓ Callbacks configurés, modèle sera sauvegardé dans: {model_dir}")

    return callbacks, model_dir


def train_model(
    use_pretrained: bool = False,
    data_dir: str = None,
    epochs: int = None,
    batch_size: int = None
):
    """
    Entraîne le modèle Vision Transformer

    Args:
        use_pretrained: Utiliser un backbone pré-entraîné
        data_dir: Répertoire des données (None = config.RAW_DATA_DIR)
        epochs: Nombre d'epochs (None = config.EPOCHS)
        batch_size: Taille du batch (None = config.BATCH_SIZE)
    """
    print("\n" + "="*70)
    print("ENTRAÎNEMENT DU MODÈLE VISION TRANSFORMER")
    print("="*70 + "\n")

    # Configuration
    setup_gpu()

    data_dir = data_dir or config.RAW_DATA_DIR
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE

    # Fixer le seed pour reproductibilité
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # =========================================================================
    # 1. Chargement des données
    # =========================================================================
    print("\n📂 Chargement des données...")

    data_loader = MammogramDataLoader(
        data_dir=data_dir,
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=batch_size,
        class_names=config.CLASS_NAMES
    )

    try:
        train_data, val_data, test_data = data_loader.load_from_directory(
            train_split=config.TRAIN_SPLIT,
            val_split=config.VAL_SPLIT,
            test_split=config.TEST_SPLIT,
            random_seed=config.RANDOM_SEED
        )
    except ValueError as e:
        print(f"\n⚠️ {e}")
        print("\n💡 Conseil: Créez un dataset d'exemple avec:")
        print("   from src.data_loader import create_sample_dataset")
        print(f"   create_sample_dataset('{data_dir}', num_samples_per_class=50)")
        return

    # Sauvegarder les infos du dataset
    dataset_info_path = os.path.join(config.DATA_DIR, 'dataset_info.json')
    data_loader.save_dataset_info(dataset_info_path, train_data, val_data, test_data)

    # Créer les datasets TensorFlow
    train_dataset = data_loader.create_tf_dataset(
        train_data[0], train_data[1],
        shuffle=True, augment=True
    )

    val_dataset = data_loader.create_tf_dataset(
        val_data[0], val_data[1],
        shuffle=False, augment=False
    )

    test_dataset = data_loader.create_tf_dataset(
        test_data[0], test_data[1],
        shuffle=False, augment=False
    )

    # =========================================================================
    # 2. Création du modèle
    # =========================================================================
    print("\n🏗️ Création du modèle...")

    if use_pretrained:
        print("  → Utilisation d'un backbone pré-entraîné")
        model = create_vit_with_pretrained_backbone(
            image_size=config.IMAGE_SIZE,
            num_classes=config.NUM_CLASSES,
            dropout_rate=config.DROPOUT_RATE
        )
    else:
        print("  → Entraînement from scratch")
        model = create_vit_model(
            image_size=config.IMAGE_SIZE,
            patch_size=config.PATCH_SIZE,
            num_classes=config.NUM_CLASSES,
            projection_dim=config.PROJECTION_DIM,
            num_heads=config.NUM_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            mlp_head_units=config.MLP_HEAD_UNITS,
            dropout_rate=config.DROPOUT_RATE
        )

    # =========================================================================
    # 3. Compilation
    # =========================================================================
    print("\n⚙️ Compilation du modèle...")

    # Learning rate schedule avec warmup
    total_steps = len(train_data[0]) // batch_size * epochs
    warmup_steps = len(train_data[0]) // batch_size * config.WARMUP_EPOCHS

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.LEARNING_RATE,
        decay_steps=total_steps - warmup_steps,
        alpha=0.1
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.WEIGHT_DECAY
    )

    # Métriques
    metrics = [
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc', multi_label=True)
    ]

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=metrics
    )

    # Afficher l'architecture
    model.summary()
    print(f"\n📊 Paramètres du modèle: {model.count_params():,}")

    # =========================================================================
    # 4. Entraînement
    # =========================================================================
    print("\n🚀 Début de l'entraînement...\n")

    callbacks, model_dir = create_callbacks("vit_breast_cancer")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Sauvegarder l'historique
    history_path = os.path.join(model_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=4)

    print(f"\n✓ Entraînement terminé!")
    print(f"✓ Modèle sauvegardé: {model_dir}")

    # =========================================================================
    # 5. Évaluation sur le test set
    # =========================================================================
    print("\n📊 Évaluation sur le test set...")

    # Prédictions
    y_true = []
    y_pred = []
    y_pred_proba = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    # Calculer les métriques
    metrics_calc = MetricsCalculator(config.CLASS_NAMES)
    metrics_dict = metrics_calc.calculate_all_metrics(y_true, y_pred, y_pred_proba)

    # Afficher le rapport
    metrics_calc.print_metrics_report(metrics_dict)

    # Sauvegarder les métriques
    metrics_path = os.path.join(config.METRICS_DIR, 'test_metrics.json')
    metrics_calc.save_metrics(metrics_dict, metrics_path)

    # Visualisations
    visualizer = MetricsVisualizer(config.CLASS_NAMES)

    # Matrice de confusion
    cm = np.array(metrics_dict['confusion_matrix'])
    cm_path = os.path.join(config.PLOTS_DIR, 'confusion_matrix.png')
    visualizer.plot_confusion_matrix(cm, save_path=cm_path)

    # Courbes ROC
    roc_path = os.path.join(config.PLOTS_DIR, 'roc_curves.png')
    visualizer.plot_roc_curves(y_true, y_pred_proba, save_path=roc_path)

    # Comparaison des métriques
    comparison_path = os.path.join(config.PLOTS_DIR, 'metrics_comparison.png')
    visualizer.plot_metrics_comparison(metrics_dict, save_path=comparison_path)

    # Historique d'entraînement
    history_plot_path = os.path.join(config.PLOTS_DIR, 'training_history.png')
    visualizer.plot_training_history(history.history, save_path=history_plot_path)

    print(f"\n✓ Évaluation terminée!")
    print(f"✓ Résultats sauvegardés dans: {config.RESULTS_DIR}")

    return model, history, metrics_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Entraîner le modèle Vision Transformer')
    parser.add_argument('--pretrained', action='store_true',
                        help='Utiliser un backbone pré-entraîné')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Répertoire des données')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Taille du batch')

    args = parser.parse_args()

    train_model(
        use_pretrained=args.pretrained,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
