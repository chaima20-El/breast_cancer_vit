"""
Script principal pour gérer l'ensemble du pipeline
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.train import train_model
from src.evaluate import (
    load_model, predict_single_image, predict_batch,
    evaluate_on_dataset, generate_gradcam_visualizations,
    print_prediction_results
)
from src.data_loader import create_sample_dataset, MammogramDataLoader


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline Vision Transformer pour le diagnostic du cancer du sein',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Créer un dataset d'exemple
  python main.py --mode create_sample --output data/raw --num_samples 50

  # Entraîner le modèle
  python main.py --mode train --epochs 50 --batch_size 32

  # Entraîner avec backbone pré-entraîné
  python main.py --mode train --pretrained --epochs 30

  # Prédire une image
  python main.py --mode predict --model models/saved_models/best_model.h5 --image data/test/image.png

  # Évaluer sur un répertoire d'images
  python main.py --mode evaluate --model models/saved_models/best_model.h5 --data_dir data/test

  # Générer des visualisations Grad-CAM
  python main.py --mode gradcam --model models/saved_models/best_model.h5 --image data/test/image.png

  # Pipeline complet: créer données + entraîner + évaluer
  python main.py --mode full_pipeline --num_samples 100 --epochs 30
        """
    )

    # Mode de fonctionnement
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['create_sample', 'train', 'predict', 'evaluate', 'gradcam', 'full_pipeline'],
        help='Mode d\'exécution'
    )

    # Arguments pour create_sample
    parser.add_argument('--output', type=str, default=config.RAW_DATA_DIR,
                        help='Répertoire de sortie pour le dataset')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Nombre d\'échantillons par classe')

    # Arguments pour train
    parser.add_argument('--pretrained', action='store_true',
                        help='Utiliser un backbone pré-entraîné')
    parser.add_argument('--data_dir', type=str, default=config.RAW_DATA_DIR,
                        help='Répertoire des données')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Taille du batch')

    # Arguments pour predict/evaluate/gradcam
    parser.add_argument('--model', type=str,
                        help='Chemin vers le modèle sauvegardé')
    parser.add_argument('--image', type=str,
                        help='Chemin d\'une image pour prédiction')
    parser.add_argument('--num_gradcam', type=int, default=10,
                        help='Nombre de visualisations Grad-CAM')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("DIAGNOSTIC DU CANCER DU SEIN - VISION TRANSFORMER")
    print("="*70 + "\n")

    # =========================================================================
    # Mode: Créer un dataset d'exemple
    # =========================================================================
    if args.mode == 'create_sample':
        print(f"📦 Création d'un dataset d'exemple...")
        create_sample_dataset(args.output, args.num_samples)

    # =========================================================================
    # Mode: Entraînement
    # =========================================================================
    elif args.mode == 'train':
        train_model(
            use_pretrained=args.pretrained,
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    # =========================================================================
    # Mode: Prédiction sur une image
    # =========================================================================
    elif args.mode == 'predict':
        if not args.model:
            print("❌ Erreur: --model requis pour le mode predict")
            sys.exit(1)
        if not args.image:
            print("❌ Erreur: --image requis pour le mode predict")
            sys.exit(1)

        model = load_model(args.model)
        results = predict_single_image(model, args.image, config.CLASS_NAMES)
        print_prediction_results(results)

    # =========================================================================
    # Mode: Évaluation sur un dataset
    # =========================================================================
    elif args.mode == 'evaluate':
        if not args.model:
            print("❌ Erreur: --model requis pour le mode evaluate")
            sys.exit(1)
        if not args.data_dir:
            print("❌ Erreur: --data_dir requis pour le mode evaluate")
            sys.exit(1)

        model = load_model(args.model)

        # Charger les données
        data_loader = MammogramDataLoader(
            data_dir=args.data_dir,
            image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            batch_size=args.batch_size,
            class_names=config.CLASS_NAMES
        )

        _, _, test_data = data_loader.load_from_directory(
            train_split=0.0,
            val_split=0.0,
            test_split=1.0,
            random_seed=config.RANDOM_SEED
        )

        # Évaluer
        evaluate_on_dataset(
            model,
            test_data[0],
            test_data[1],
            config.CLASS_NAMES,
            config.METRICS_DIR
        )

    # =========================================================================
    # Mode: Visualisations Grad-CAM
    # =========================================================================
    elif args.mode == 'gradcam':
        if not args.model:
            print("❌ Erreur: --model requis pour le mode gradcam")
            sys.exit(1)

        model = load_model(args.model)

        if args.image:
            # Une seule image
            generate_gradcam_visualizations(
                model, [args.image], config.CLASS_NAMES,
                config.GRADCAM_DIR, num_images=1
            )
        elif args.data_dir:
            # Toutes les images du répertoire
            from pathlib import Path
            image_paths = []
            for ext in ['.png', '.jpg', '.jpeg']:
                image_paths.extend(list(Path(args.data_dir).glob(f'**/*{ext}')))
            image_paths = [str(p) for p in image_paths]

            generate_gradcam_visualizations(
                model, image_paths, config.CLASS_NAMES,
                config.GRADCAM_DIR, num_images=args.num_gradcam
            )
        else:
            print("❌ Erreur: --image ou --data_dir requis pour le mode gradcam")
            sys.exit(1)

    # =========================================================================
    # Mode: Pipeline complet
    # =========================================================================
    elif args.mode == 'full_pipeline':
        print("🚀 Exécution du pipeline complet...\n")

        # 1. Créer le dataset
        print("\n[1/3] Création du dataset d'exemple...")
        create_sample_dataset(args.output, args.num_samples)

        # 2. Entraîner le modèle
        print("\n[2/3] Entraînement du modèle...")
        model, history, metrics = train_model(
            use_pretrained=args.pretrained,
            data_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        # 3. Générer des visualisations Grad-CAM
        print("\n[3/3] Génération de visualisations Grad-CAM...")
        from pathlib import Path
        image_paths = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_paths.extend(list(Path(args.output).glob(f'**/*{ext}')))
        image_paths = [str(p) for p in image_paths[:args.num_gradcam]]

        if len(image_paths) > 0:
            generate_gradcam_visualizations(
                model, image_paths, config.CLASS_NAMES,
                config.GRADCAM_DIR, num_images=min(10, len(image_paths))
            )

        print("\n✅ Pipeline complet terminé avec succès!")
        print(f"\n📊 Résumé:")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"  - AUC-ROC (Macro): {metrics.get('auc_roc_macro', 'N/A')}")
        print(f"\n📂 Résultats disponibles dans: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
