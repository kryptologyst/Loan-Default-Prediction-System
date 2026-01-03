"""Main training script for loan default prediction."""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import LoanDataGenerator, LoanDataConfig, LoanDataPreprocessor
from features import LoanFeatureEngineer
from models import (
    RandomForestModel, XGBoostModel, LightGBMModel, 
    LogisticRegressionModel, ModelEnsemble
)
from evaluation import ModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create necessary directories for the project."""
    directories = [
        "data", "configs", "scripts", "notebooks", 
        "tests", "assets", "demo", "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")


def generate_data(config: LoanDataConfig) -> pd.DataFrame:
    """Generate synthetic loan data.
    
    Args:
        config: Configuration for data generation.
        
    Returns:
        Generated loan data DataFrame.
    """
    logger.info("Generating synthetic loan data...")
    generator = LoanDataGenerator(config)
    df = generator.generate_features()
    
    # Save raw data
    df.to_csv("data/raw_loan_data.csv", index=False)
    logger.info(f"Saved raw data to data/raw_loan_data.csv ({len(df)} samples)")
    
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the loan data.
    
    Args:
        df: Raw loan data DataFrame.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor).
    """
    logger.info("Preprocessing loan data...")
    
    # Feature engineering
    feature_engineer = LoanFeatureEngineer()
    df_engineered = feature_engineer.engineer_all_features(df)
    
    # Save engineered data
    df_engineered.to_csv("data/engineered_loan_data.csv", index=False)
    logger.info(f"Saved engineered data to data/engineered_loan_data.csv")
    
    # Preprocessing
    preprocessor = LoanDataPreprocessor()
    X, y = preprocessor.fit_transform(df_engineered)
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Default rate in train: {y_train.mean():.2%}")
    logger.info(f"Default rate in test: {y_test.mean():.2%}")
    
    # Save preprocessor
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    logger.info("Saved preprocessor to models/preprocessor.pkl")
    
    return X_train, X_test, y_train, y_test, preprocessor


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """Train multiple models for loan default prediction.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        
    Returns:
        Dictionary of trained models.
    """
    logger.info("Training multiple models...")
    
    models = {
        'random_forest': RandomForestModel(n_estimators=100, random_state=42),
        'xgboost': XGBoostModel(n_estimators=100, random_state=42),
        'lightgbm': LightGBMModel(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegressionModel(random_state=42)
    }
    
    # Train each model
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Save model
        model.save(f"models/{name}_model.pkl")
        logger.info(f"Saved {name} model")
    
    # Create ensemble
    ensemble_models = [models['random_forest'], models['xgboost'], models['lightgbm']]
    ensemble = ModelEnsemble(ensemble_models, weights=[0.3, 0.4, 0.3])
    ensemble.fit(X_train, y_train)
    ensemble.save("models/ensemble_model.pkl")
    
    models['ensemble'] = ensemble
    logger.info("Created and trained ensemble model")
    
    return models


def evaluate_models(models: Dict[str, Any], X_test: np.ndarray, 
                   y_test: np.ndarray) -> ModelEvaluator:
    """Evaluate all trained models.
    
    Args:
        models: Dictionary of trained models.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        ModelEvaluator with evaluation results.
    """
    logger.info("Evaluating models...")
    
    evaluator = ModelEvaluator()
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Evaluate
        metrics = evaluator.evaluate_model(y_test, y_pred, y_prob, name)
        
        # Generate plots
        if y_prob is not None:
            evaluator.plot_roc_curve(y_test, y_prob, name, f"assets/{name}_roc.png")
            evaluator.plot_precision_recall_curve(y_test, y_prob, name, f"assets/{name}_pr.png")
            evaluator.plot_calibration_curve(y_test, y_prob, name, save_path=f"assets/{name}_calibration.png")
        
        evaluator.plot_confusion_matrix(y_test, y_pred, name, f"assets/{name}_confusion.png")
        
        # Feature importance for tree-based models
        if hasattr(model, 'get_feature_importance'):
            try:
                importance = model.get_feature_importance()
                feature_names = [f"feature_{i}" for i in range(len(importance))]
                evaluator.plot_feature_importance(feature_names, importance, name, save_path=f"assets/{name}_importance.png")
            except Exception as e:
                logger.warning(f"Could not plot feature importance for {name}: {e}")
    
    # Generate summary report
    summary_df = evaluator.get_metrics_summary()
    summary_df.to_csv("assets/model_comparison.csv", index=False)
    logger.info("Saved model comparison to assets/model_comparison.csv")
    
    return evaluator


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train loan default prediction models")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    
    logger.info("Starting loan default prediction training pipeline")
    logger.info(f"Configuration: {args.n_samples} samples, seed={args.random_seed}")
    
    # Setup
    setup_directories()
    
    # Data generation
    config = LoanDataConfig(n_samples=args.n_samples, random_seed=args.random_seed)
    df = generate_data(config)
    
    # Preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Model training
    models = train_models(X_train, y_train)
    
    # Evaluation
    evaluator = evaluate_models(models, X_test, y_test)
    
    # Print summary
    summary_df = evaluator.get_metrics_summary()
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(summary_df[['model_name', 'auc_roc', 'gini', 'ks_statistic', 'accuracy', 'f1_score']].to_string(index=False))
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
