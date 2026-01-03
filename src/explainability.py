"""SHAP explainability utilities for loan default prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based explainability for loan default prediction models."""
    
    def __init__(self, model: Any, feature_names: List[str]):
        """Initialize SHAP explainer.
        
        Args:
            model: Trained model object.
            feature_names: List of feature names.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explainability. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_background: np.ndarray, model_type: str = 'tree') -> None:
        """Create SHAP explainer for the model.
        
        Args:
            X_background: Background dataset for explainer.
            model_type: Type of model ('tree', 'linear', 'deep', etc.).
        """
        logger.info(f"Creating SHAP explainer for {model_type} model")
        
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif model_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_background)
        else:
            # Use KernelExplainer as fallback
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
        
        logger.info("SHAP explainer created successfully")
    
    def explain_predictions(self, X: np.ndarray) -> np.ndarray:
        """Generate SHAP values for predictions.
        
        Args:
            X: Feature matrix to explain.
            
        Returns:
            SHAP values array.
        """
        if self.explainer is None:
            raise ValueError("Explainer must be created before explaining predictions")
        
        logger.info(f"Generating SHAP values for {len(X)} samples")
        
        try:
            self.shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class case
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Use positive class
            
            logger.info("SHAP values generated successfully")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"Error generating SHAP values: {e}")
            raise
    
    def plot_summary(self, X: np.ndarray, max_display: int = 20, 
                    save_path: Optional[str] = None) -> None:
        """Plot SHAP summary plot.
        
        Args:
            X: Feature matrix.
            max_display: Maximum number of features to display.
            save_path: Optional path to save the plot.
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, 
                         max_display=max_display, show=False)
        plt.title('SHAP Summary Plot')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_waterfall(self, X: np.ndarray, sample_idx: int = 0,
                      save_path: Optional[str] = None) -> None:
        """Plot SHAP waterfall plot for a single prediction.
        
        Args:
            X: Feature matrix.
            sample_idx: Index of sample to explain.
            save_path: Optional path to save the plot.
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            self.explainer.expected_value,
            self.shap_values[sample_idx],
            X[sample_idx],
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, save_path: Optional[str] = None) -> None:
        """Plot SHAP feature importance.
        
        Args:
            save_path: Optional path to save the plot.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be generated first")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Sort features by importance
        sorted_indices = np.argsort(mean_shap)[::-1]
        sorted_features = [self.feature_names[i] for i in sorted_indices]
        sorted_values = mean_shap[sorted_indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_features)), sorted_values)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Mean |SHAP value|')
        plt.title('SHAP Feature Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dependence(self, X: np.ndarray, feature_idx: int,
                       save_path: Optional[str] = None) -> None:
        """Plot SHAP dependence plot for a specific feature.
        
        Args:
            X: Feature matrix.
            feature_idx: Index of feature to plot.
            save_path: Optional path to save the plot.
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {self.feature_names[feature_idx]}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """Get feature importance as DataFrame.
        
        Returns:
            DataFrame with feature names and importance scores.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be generated first")
        
        mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
    
    def explain_single_prediction(self, X: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain a single prediction in detail.
        
        Args:
            X: Feature matrix.
            sample_idx: Index of sample to explain.
            
        Returns:
            Dictionary with explanation details.
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(X[sample_idx:sample_idx+1])[0]
        else:
            prediction = self.model.predict(X[sample_idx:sample_idx+1])[0]
        
        # Get SHAP values for this sample
        sample_shap = self.shap_values[sample_idx]
        sample_features = X[sample_idx]
        
        # Create explanation
        explanation = {
            'prediction': prediction,
            'expected_value': self.explainer.expected_value,
            'feature_contributions': []
        }
        
        # Sort features by absolute SHAP value
        sorted_indices = np.argsort(np.abs(sample_shap))[::-1]
        
        for idx in sorted_indices:
            explanation['feature_contributions'].append({
                'feature': self.feature_names[idx],
                'value': sample_features[idx],
                'shap_value': sample_shap[idx],
                'contribution': sample_shap[idx]
            })
        
        return explanation


class ModelExplainer:
    """High-level model explainer that works with different model types."""
    
    def __init__(self, models: Dict[str, Any], feature_names: List[str]):
        """Initialize model explainer.
        
        Args:
            models: Dictionary of trained models.
            feature_names: List of feature names.
        """
        self.models = models
        self.feature_names = feature_names
        self.explainers = {}
        
    def create_explainers(self, X_background: np.ndarray) -> None:
        """Create SHAP explainers for all models.
        
        Args:
            X_background: Background dataset for explainers.
        """
        for name, model in self.models.items():
            try:
                explainer = SHAPExplainer(model, self.feature_names)
                
                # Determine model type
                if 'random_forest' in name.lower() or 'xgboost' in name.lower() or 'lightgbm' in name.lower():
                    model_type = 'tree'
                elif 'logistic' in name.lower():
                    model_type = 'linear'
                else:
                    model_type = 'tree'  # Default fallback
                
                explainer.create_explainer(X_background, model_type)
                self.explainers[name] = explainer
                
                logger.info(f"Created explainer for {name}")
                
            except Exception as e:
                logger.warning(f"Could not create explainer for {name}: {e}")
    
    def compare_model_explanations(self, X: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """Compare explanations across different models.
        
        Args:
            X: Feature matrix.
            sample_idx: Index of sample to explain.
            
        Returns:
            Dictionary with explanations from all models.
        """
        explanations = {}
        
        for name, explainer in self.explainers.items():
            try:
                explanations[name] = explainer.explain_single_prediction(X, sample_idx)
            except Exception as e:
                logger.warning(f"Could not explain {name}: {e}")
        
        return explanations
