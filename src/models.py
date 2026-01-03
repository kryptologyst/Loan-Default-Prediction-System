"""Machine learning models for loan default prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all loan default prediction models."""
    
    def __init__(self, name: str):
        """Initialize the base model.
        
        Args:
            name: Name of the model.
        """
        self.name = name
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Fit the model to training data.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            **kwargs: Additional arguments.
            
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted labels.
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted probabilities.
        """
        pass
    
    def save(self, filepath: str) -> None:
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model {self.name} saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the model from disk.
        
        Args:
            filepath: Path to load the model from.
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model {self.name} loaded from {filepath}")


class RandomForestModel(BaseModel):
    """Random Forest model for loan default prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 random_state: int = 42, **kwargs):
        """Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees.
            random_state: Random state for reproducibility.
            **kwargs: Additional arguments for RandomForestClassifier.
        """
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForestModel':
        """Fit the Random Forest model.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            **kwargs: Additional arguments.
            
        Returns:
            Self for method chaining.
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        logger.info(f"Random Forest model fitted with {self.model.n_estimators} trees")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.
        
        Returns:
            Feature importance array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class XGBoostModel(BaseModel):
    """XGBoost model for loan default prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42, **kwargs):
        """Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum depth of trees.
            learning_rate: Learning rate.
            random_state: Random state for reproducibility.
            **kwargs: Additional arguments for XGBClassifier.
        """
        super().__init__("XGBoost")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='logloss',
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'XGBoostModel':
        """Fit the XGBoost model.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            **kwargs: Additional arguments.
            
        Returns:
            Self for method chaining.
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        logger.info(f"XGBoost model fitted with {self.model.n_estimators} rounds")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.
        
        Returns:
            Feature importance array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class LightGBMModel(BaseModel):
    """LightGBM model for loan default prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42, **kwargs):
        """Initialize LightGBM model.
        
        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum depth of trees.
            learning_rate: Learning rate.
            random_state: Random state for reproducibility.
            **kwargs: Additional arguments for LGBMClassifier.
        """
        super().__init__("LightGBM")
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=-1,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LightGBMModel':
        """Fit the LightGBM model.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            **kwargs: Additional arguments.
            
        Returns:
            Self for method chaining.
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        logger.info(f"LightGBM model fitted with {self.model.n_estimators} rounds")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.
        
        Returns:
            Feature importance array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for loan default prediction."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        """Initialize Logistic Regression model.
        
        Args:
            random_state: Random state for reproducibility.
            **kwargs: Additional arguments for LogisticRegression.
        """
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LogisticRegressionModel':
        """Fit the Logistic Regression model.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            **kwargs: Additional arguments.
            
        Returns:
            Self for method chaining.
        """
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        logger.info("Logistic Regression model fitted")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores (coefficients).
        
        Returns:
            Feature importance array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return np.abs(self.model.coef_[0])


class ModelEnsemble:
    """Ensemble of multiple models for loan default prediction."""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        """Initialize the ensemble.
        
        Args:
            models: List of models to ensemble.
            weights: Optional weights for each model. If None, equal weights are used.
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.is_fitted = False
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'ModelEnsemble':
        """Fit all models in the ensemble.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            **kwargs: Additional arguments.
            
        Returns:
            Self for method chaining.
        """
        for model in self.models:
            model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        logger.info(f"Ensemble fitted with {len(self.models)} models")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted labels.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get probability predictions from each model
        proba_predictions = []
        for model in self.models:
            proba = model.predict_proba(X)[:, 1]  # Probability of default
            proba_predictions.append(proba)
        
        # Weighted average of probabilities
        ensemble_proba = np.average(proba_predictions, axis=0, weights=self.weights)
        
        # Convert to binary predictions
        return (ensemble_proba > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble class probabilities.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get probability predictions from each model
        proba_predictions = []
        for model in self.models:
            proba = model.predict_proba(X)
            proba_predictions.append(proba)
        
        # Weighted average of probabilities
        ensemble_proba = np.average(proba_predictions, axis=0, weights=self.weights)
        
        return ensemble_proba
