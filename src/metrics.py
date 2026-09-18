"""
Métriques d'évaluation complètes pour le diagnostic du cancer du sein
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
import pandas as pd


class MetricsCalculator:
    """Calcule toutes les métriques d'évaluation"""

    def __init__(self, class_names: List[str]):
        """
        Args:
            class_names: Liste des noms de classes
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict:
        """
        Calcule toutes les métriques

        Args:
            y_true: Labels vrais (indices de classes)
            y_pred: Prédictions (indices de classes)
            y_pred_proba: Probabilités prédites (optionnel)

        Returns:
            Dictionnaire de métriques
        """
        metrics = {}

        # Métriques de base
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Métriques par classe et moyennes
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()

        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()

        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

        # Sensibilité (Recall) et Spécificité
        cm = confusion_matrix(y_true, y_pred)
        sensitivity_per_class, specificity_per_class = self._calculate_sensitivity_specificity(cm)
        metrics['sensitivity_per_class'] = sensitivity_per_class.tolist()
        metrics['specificity_per_class'] = specificity_per_class.tolist()
        metrics['sensitivity_macro'] = np.mean(sensitivity_per_class)
        metrics['specificity_macro'] = np.mean(specificity_per_class)

        # Cohen's Kappa
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Matthews Correlation Coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        # AUC-ROC (si probabilités disponibles)
        if y_pred_proba is not None:
            try:
                # One-vs-Rest AUC
                if self.num_classes == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['auc_roc_macro'] = roc_auc_score(
                        y_true, y_pred_proba,
                        multi_class='ovr', average='macro'
                    )
                    metrics['auc_roc_weighted'] = roc_auc_score(
                        y_true, y_pred_proba,
                        multi_class='ovr', average='weighted'
                    )
                    metrics['auc_roc_per_class'] = roc_auc_score(
                        y_true, y_pred_proba,
                        multi_class='ovr', average=None
                    ).tolist()
            except ValueError as e:
                print(f"Impossible de calculer AUC-ROC: {e}")

        # Matrice de confusion
        metrics['confusion_matrix'] = cm.tolist()

        return metrics

    def _calculate_sensitivity_specificity(
        self,
        cm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule sensibilité et spécificité à partir de la matrice de confusion

        Args:
            cm: Matrice de confusion

        Returns:
            (sensitivity, specificity) pour chaque classe
        """
        n_classes = cm.shape[0]
        sensitivity = np.zeros(n_classes)
        specificity = np.zeros(n_classes)

        for i in range(n_classes):
            # True Positives, False Negatives
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp

            # True Negatives, False Positives
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + tp
            fp = np.sum(cm[:, i]) - tp

            # Sensibilité (Recall/TPR)
            sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Spécificité (TNR)
            specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return sensitivity, specificity

    def print_metrics_report(self, metrics: Dict):
        """
        Affiche un rapport formaté des métriques

        Args:
            metrics: Dictionnaire de métriques
        """
        print("\n" + "=" * 70)
        print("RAPPORT D'ÉVALUATION - DIAGNOSTIC DU CANCER DU SEIN")
        print("=" * 70)

        print(f"\n📊 MÉTRIQUES GLOBALES:")
        print(f"  Accuracy                : {metrics['accuracy']:.4f}")
        print(f"  Precision (Macro)       : {metrics['precision_macro']:.4f}")
        print(f"  Recall/Sensitivity (Macro): {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (Macro)        : {metrics['f1_macro']:.4f}")
        print(f"  Specificity (Macro)     : {metrics['specificity_macro']:.4f}")
        print(f"  Cohen's Kappa           : {metrics['cohen_kappa']:.4f}")
        print(f"  Matthews Corr. Coef.    : {metrics['mcc']:.4f}")

        if 'auc_roc_macro' in metrics:
            print(f"  AUC-ROC (Macro)         : {metrics['auc_roc_macro']:.4f}")

        print(f"\n📈 MÉTRIQUES PAR CLASSE:")
        for i, class_name in enumerate(self.class_names):
            print(f"\n  {class_name.upper()}:")
            print(f"    Precision   : {metrics['precision_per_class'][i]:.4f}")
            print(f"    Recall      : {metrics['recall_per_class'][i]:.4f}")
            print(f"    F1-Score    : {metrics['f1_per_class'][i]:.4f}")
            print(f"    Sensitivity : {metrics['sensitivity_per_class'][i]:.4f}")
            print(f"    Specificity : {metrics['specificity_per_class'][i]:.4f}")
            if 'auc_roc_per_class' in metrics:
                print(f"    AUC-ROC     : {metrics['auc_roc_per_class'][i]:.4f}")

        print("\n" + "=" * 70)

    def save_metrics(self, metrics: Dict, filepath: str):
        """
        Sauvegarde les métriques en JSON

        Args:
            metrics: Dictionnaire de métriques
            filepath: Chemin du fichier de sortie
        """
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Métriques sauvegardées: {filepath}")


class MetricsVisualizer:
    """Crée des visualisations pour les métriques"""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: str = None,
        normalize: bool = False
    ):
        """
        Affiche la matrice de confusion

        Args:
            cm: Matrice de confusion
            save_path: Chemin pour sauvegarder la figure
            normalize: Normaliser la matrice
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Matrice de Confusion Normalisée'
        else:
            fmt = 'd'
            title = 'Matrice de Confusion'

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Nombre'}
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('Vraie Classe', fontsize=12)
        plt.xlabel('Classe Prédite', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matrice de confusion sauvegardée: {save_path}")

        plt.show()

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = None
    ):
        """
        Affiche les courbes ROC pour chaque classe

        Args:
            y_true: Labels vrais
            y_pred_proba: Probabilités prédites
            save_path: Chemin pour sauvegarder
        """
        from sklearn.preprocessing import label_binarize

        # Binariser les labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        plt.figure(figsize=(10, 8))

        # ROC pour chaque classe
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})',
                linewidth=2
            )

        # Ligne diagonale
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=12)
        plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=12)
        plt.title('Courbes ROC - Classification Multi-Classes', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Courbes ROC sauvegardées: {save_path}")

        plt.show()

    def plot_metrics_comparison(
        self,
        metrics: Dict,
        save_path: str = None
    ):
        """
        Compare les métriques par classe

        Args:
            metrics: Dictionnaire de métriques
            save_path: Chemin pour sauvegarder
        """
        metrics_to_plot = ['precision_per_class', 'recall_per_class',
                          'f1_per_class', 'sensitivity_per_class',
                          'specificity_per_class']

        labels = ['Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity']

        data = []
        for metric_key in metrics_to_plot:
            if metric_key in metrics:
                data.append(metrics[metric_key])

        x = np.arange(len(self.class_names))
        width = 0.15

        fig, ax = plt.subplots(figsize=(14, 8))

        for i, (values, label) in enumerate(zip(data, labels)):
            offset = width * (i - len(data) / 2)
            ax.bar(x + offset, values, width, label=label)

        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comparaison des Métriques par Classe', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparaison des métriques sauvegardée: {save_path}")

        plt.show()

    def plot_training_history(
        self,
        history: dict,
        save_path: str = None
    ):
        """
        Affiche l'historique d'entraînement

        Args:
            history: Historique retourné par model.fit()
            save_path: Chemin pour sauvegarder
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Train Precision', linewidth=2)
            axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2)
            axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

        # Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2)
            axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
            axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Historique d'entraînement sauvegardé: {save_path}")

        plt.show()


if __name__ == "__main__":
    print("Module de métriques chargé avec succès")
    print("\nClasses disponibles:")
    print("- MetricsCalculator")
    print("- MetricsVisualizer")
