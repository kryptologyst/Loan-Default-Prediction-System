"""Evaluation metrics and utilities for loan default prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class CreditMetrics:
    """Credit-specific evaluation metrics for loan default prediction."""
    
    @staticmethod
    def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calculate Gini coefficient.
        
        Args:
            y_true: True binary labels.
            y_score: Predicted probabilities or scores.
            
        Returns:
            Gini coefficient.
        """
        auc = roc_auc_score(y_true, y_score)
        return 2 * auc - 1
    
    @staticmethod
    def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic.
        
        Args:
            y_true: True binary labels.
            y_score: Predicted probabilities or scores.
            
        Returns:
            KS statistic.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        return np.max(tpr - fpr)
    
    @staticmethod
    def population_stability_index(y_true: np.ndarray, y_score: np.ndarray, 
                                 bins: int = 10) -> float:
        """Calculate Population Stability Index.
        
        Args:
            y_true: True binary labels.
            y_score: Predicted probabilities or scores.
            bins: Number of bins for PSI calculation.
            
        Returns:
            PSI value.
        """
        # Create bins
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Calculate distributions
        expected_dist = np.histogram(y_score, bins=bin_edges)[0] / len(y_score)
        actual_dist = np.histogram(y_score, bins=bin_edges)[0] / len(y_score)
        
        # Calculate PSI
        psi = 0
        for i in range(len(expected_dist)):
            if expected_dist[i] > 0 and actual_dist[i] > 0:
                psi += (actual_dist[i] - expected_dist[i]) * np.log(actual_dist[i] / expected_dist[i])
        
        return psi
    
    @staticmethod
    def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
        """Calculate calibration error.
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities.
            bins: Number of bins for calibration calculation.
            
        Returns:
            Calibration error.
        """
        bin_boundaries = np.linspace(0, 1, bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class ModelEvaluator:
    """Comprehensive model evaluation for loan default prediction."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.metrics_history = []
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_prob: Optional[np.ndarray] = None, 
                      model_name: str = "Model") -> Dict[str, float]:
        """Evaluate model performance comprehensively.
        
        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            y_prob: Predicted probabilities.
            model_name: Name of the model for logging.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        if y_prob is not None:
            # Probability-based metrics
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
            metrics['brier_score'] = brier_score_loss(y_true, y_prob)
            
            # Credit-specific metrics
            metrics['gini'] = CreditMetrics.gini_coefficient(y_true, y_prob)
            metrics['ks_statistic'] = CreditMetrics.ks_statistic(y_true, y_prob)
            metrics['calibration_error'] = CreditMetrics.calibration_error(y_true, y_prob)
        
        # Store metrics
        metrics['model_name'] = model_name
        self.metrics_history.append(metrics.copy())
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}")
        logger.info(f"Gini: {metrics.get('gini', 'N/A'):.4f}")
        logger.info(f"KS: {metrics.get('ks_statistic', 'N/A'):.4f}")
        
        return metrics
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      model_name: str = "Model", save_path: Optional[str] = None) -> None:
        """Plot ROC curve.
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities.
            model_name: Name of the model.
            save_path: Optional path to save the plot.
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   model_name: str = "Model", save_path: Optional[str] = None) -> None:
        """Plot Precision-Recall curve.
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities.
            model_name: Name of the model.
            save_path: Optional path to save the plot.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {auc_pr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                              model_name: str = "Model", bins: int = 10,
                              save_path: Optional[str] = None) -> None:
        """Plot calibration curve.
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities.
            model_name: Name of the model.
            bins: Number of bins for calibration.
            save_path: Optional path to save the plot.
        """
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=bins
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f'{model_name}')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str = "Model", save_path: Optional[str] = None) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            model_name: Name of the model.
            save_path: Optional path to save the plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_scores: np.ndarray,
                               model_name: str = "Model", top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
        """Plot feature importance.
        
        Args:
            feature_names: List of feature names.
            importance_scores: Feature importance scores.
            model_name: Name of the model.
            top_n: Number of top features to show.
            save_path: Optional path to save the plot.
        """
        # Get top N features
        top_indices = np.argsort(importance_scores)[-top_n:]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = importance_scores[top_indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_scores)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_prob: Optional[np.ndarray] = None,
                       model_name: str = "Model") -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            y_prob: Predicted probabilities.
            model_name: Name of the model.
            
        Returns:
            Formatted evaluation report.
        """
        metrics = self.evaluate_model(y_true, y_pred, y_prob, model_name)
        
        report = f"""
=== {model_name} Evaluation Report ===

Classification Metrics:
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}

"""
        
        if y_prob is not None:
            report += f"""Credit-Specific Metrics:
- AUC-ROC: {metrics['auc_roc']:.4f}
- AUC-PR: {metrics['auc_pr']:.4f}
- Gini Coefficient: {metrics['gini']:.4f}
- KS Statistic: {metrics['ks_statistic']:.4f}
- Brier Score: {metrics['brier_score']:.4f}
- Calibration Error: {metrics['calibration_error']:.4f}

"""
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        report += f"""Confusion Matrix:
                Predicted
                0      1
Actual    0    {cm[0,0]:4d}  {cm[0,1]:4d}
          1    {cm[1,0]:4d}  {cm[1,1]:4d}

"""
        
        return report
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of all evaluated models.
        
        Returns:
            DataFrame with metrics for all models.
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
