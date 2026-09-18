"""
Script d'évaluation du modèle sur de nouvelles données
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Optional, List
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.preprocessing import preprocess_mammogram, preprocess_batch
from src.metrics import MetricsCalculator, MetricsVisualizer
from src.gradcam import GradCAM


def load_model(model_path: str) -> keras.Model:
    """
    Charge un modèle sauvegardé

    Args:
        model_path: Chemin vers le modèle

    Returns:
        Modèle Keras
    """
    print(f"📥 Chargement du modèle: {model_path}")
    model = keras.models.load_model(model_path, compile=True)
    print(f"✓ Modèle chargé avec succès")
    return model


def predict_single_image(
    model: keras.Model,
    image_path: str,
    class_names: List[str] = None
) -> dict:
    """
    Prédit la classe d'une seule image

    Args:
        model: Modèle Keras
        image_path: Chemin de l'image
        class_names: Noms des classes

    Returns:
        Dictionnaire avec les résultats
    """
    class_names = class_names or config.CLASS_NAMES

    # Prétraiter l'image
    image = preprocess_mammogram(
        image_path,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)
    )

    # Prédiction
    image_batch = np.expand_dims(image, axis=0)
    predictions = model.predict(image_batch, verbose=0)[0]

    # Résultats
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx]

    results = {
        'image_path': image_path,
        'predicted_class': predicted_class,
        'predicted_class_idx': int(predicted_class_idx),
        'confidence': float(confidence),
        'all_probabilities': {
            class_names[i]: float(predictions[i])
            for i in range(len(class_names))
        }
    }

    return results


def predict_batch(
    model: keras.Model,
    image_paths: List[str],
    class_names: List[str] = None,
    batch_size: int = 32
) -> List[dict]:
    """
    Prédit les classes pour un batch d'images

    Args:
        model: Modèle Keras
        image_paths: Liste des chemins d'images
        class_names: Noms des classes
        batch_size: Taille du batch

    Returns:
        Liste de dictionnaires de résultats
    """
    class_names = class_names or config.CLASS_NAMES
    all_results = []

    # Traiter par batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        # Prétraiter le batch
        images = preprocess_batch(
            batch_paths,
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)
        )

        # Prédictions
        predictions = model.predict(images, verbose=0)

        # Résultats pour chaque image
        for j, path in enumerate(batch_paths):
            predicted_class_idx = np.argmax(predictions[j])
            results = {
                'image_path': path,
                'predicted_class': class_names[predicted_class_idx],
                'predicted_class_idx': int(predicted_class_idx),
                'confidence': float(predictions[j][predicted_class_idx]),
                'all_probabilities': {
                    class_names[k]: float(predictions[j][k])
                    for k in range(len(class_names))
                }
            }
            all_results.append(results)

        print(f"Traité: {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    return all_results


def evaluate_on_dataset(
    model: keras.Model,
    image_paths: np.ndarray,
    true_labels: np.ndarray,
    class_names: List[str] = None,
    output_dir: str = None
) -> dict:
    """
    Évalue le modèle sur un dataset complet

    Args:
        model: Modèle Keras
        image_paths: Chemins des images
        true_labels: Labels vrais
        class_names: Noms des classes
        output_dir: Répertoire de sortie

    Returns:
        Dictionnaire de métriques
    """
    class_names = class_names or config.CLASS_NAMES
    output_dir = output_dir or config.METRICS_DIR

    print(f"\n📊 Évaluation sur {len(image_paths)} images...")

    # Prédictions
    predictions = predict_batch(model, image_paths.tolist(), class_names)

    y_pred = np.array([p['predicted_class_idx'] for p in predictions])
    y_pred_proba = np.array([
        [p['all_probabilities'][cn] for cn in class_names]
        for p in predictions
    ])

    # Calculer les métriques
    metrics_calc = MetricsCalculator(class_names)
    metrics = metrics_calc.calculate_all_metrics(true_labels, y_pred, y_pred_proba)

    # Afficher le rapport
    metrics_calc.print_metrics_report(metrics)

    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    metrics_calc.save_metrics(metrics, metrics_path)

    # Visualisations
    visualizer = MetricsVisualizer(class_names)

    cm = np.array(metrics['confusion_matrix'])
    cm_path = os.path.join(output_dir, 'confusion_matrix_eval.png')
    visualizer.plot_confusion_matrix(cm, save_path=cm_path)

    roc_path = os.path.join(output_dir, 'roc_curves_eval.png')
    visualizer.plot_roc_curves(true_labels, y_pred_proba, save_path=roc_path)

    comparison_path = os.path.join(output_dir, 'metrics_comparison_eval.png')
    visualizer.plot_metrics_comparison(metrics, save_path=comparison_path)

    print(f"\n✓ Résultats sauvegardés dans: {output_dir}")

    return metrics


def generate_gradcam_visualizations(
    model: keras.Model,
    image_paths: List[str],
    class_names: List[str] = None,
    output_dir: str = None,
    num_images: int = 10
):
    """
    Génère des visualisations Grad-CAM

    Args:
        model: Modèle Keras
        image_paths: Chemins des images
        class_names: Noms des classes
        output_dir: Répertoire de sortie
        num_images: Nombre d'images à visualiser
    """
    class_names = class_names or config.CLASS_NAMES
    output_dir = output_dir or config.GRADCAM_DIR

    print(f"\n🔍 Génération de visualisations Grad-CAM...")

    os.makedirs(output_dir, exist_ok=True)

    # Créer l'objet Grad-CAM
    gradcam = GradCAM(model)

    # Sélectionner des images aléatoires
    if len(image_paths) > num_images:
        indices = np.random.choice(len(image_paths), num_images, replace=False)
        selected_paths = [image_paths[i] for i in indices]
    else:
        selected_paths = image_paths[:num_images]

    # Générer les visualisations
    for i, image_path in enumerate(selected_paths):
        try:
            # Prétraiter l'image
            image = preprocess_mammogram(
                image_path,
                target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)
            )

            # Visualiser
            save_path = os.path.join(output_dir, f'gradcam_{i:03d}.png')
            gradcam.visualize(image, class_names, save_path=save_path)

            print(f"  ✓ Traité: {i+1}/{len(selected_paths)}")

        except Exception as e:
            print(f"  ✗ Erreur pour {image_path}: {e}")

    print(f"\n✓ Visualisations Grad-CAM sauvegardées dans: {output_dir}")


def print_prediction_results(results: dict):
    """
    Affiche les résultats de prédiction de manière formatée

    Args:
        results: Dictionnaire de résultats
    """
    print("\n" + "="*60)
    print("RÉSULTATS DE PRÉDICTION")
    print("="*60)
    print(f"\n📄 Image: {results['image_path']}")
    print(f"\n🎯 Prédiction: {results['predicted_class'].upper()}")
    print(f"   Confiance: {results['confidence']:.2%}")
    print(f"\n📊 Probabilités pour toutes les classes:")
    for class_name, prob in results['all_probabilities'].items():
        bar = "█" * int(prob * 40)
        print(f"   {class_name:12s} {prob:6.2%} {bar}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Évaluer le modèle Vision Transformer')
    parser.add_argument('--model', type=str, required=True,
                        help='Chemin vers le modèle sauvegardé')
    parser.add_argument('--image', type=str,
                        help='Chemin d\'une seule image à prédire')
    parser.add_argument('--image_dir', type=str,
                        help='Répertoire contenant des images')
    parser.add_argument('--gradcam', action='store_true',
                        help='Générer des visualisations Grad-CAM')
    parser.add_argument('--num_gradcam', type=int, default=10,
                        help='Nombre de visualisations Grad-CAM')

    args = parser.parse_args()

    # Charger le modèle
    model = load_model(args.model)

    if args.image:
        # Prédire une seule image
        results = predict_single_image(model, args.image, config.CLASS_NAMES)
        print_prediction_results(results)

        if args.gradcam:
            generate_gradcam_visualizations(
                model, [args.image], config.CLASS_NAMES,
                num_images=1
            )

    elif args.image_dir:
        # Prédire toutes les images du répertoire
        from pathlib import Path
        image_paths = []
        for ext in ['.png', '.jpg', '.jpeg', '.dcm']:
            image_paths.extend(list(Path(args.image_dir).glob(f'**/*{ext}')))

        image_paths = [str(p) for p in image_paths]
        print(f"✓ {len(image_paths)} images trouvées dans {args.image_dir}")

        results = predict_batch(model, image_paths, config.CLASS_NAMES)

        # Sauvegarder les résultats
        output_path = os.path.join(config.RESULTS_DIR, 'predictions.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n✓ Résultats sauvegardés: {output_path}")

        if args.gradcam:
            generate_gradcam_visualizations(
                model, image_paths, config.CLASS_NAMES,
                num_images=args.num_gradcam
            )

    else:
        print("⚠️ Veuillez spécifier --image ou --image_dir")
