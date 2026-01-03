#!/usr/bin/env python3
"""Utility script for loan default prediction project."""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import LoanDataGenerator, LoanDataConfig
from models import RandomForestModel, XGBoostModel, LightGBMModel, LogisticRegressionModel
from evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_data(args):
    """Generate synthetic loan data."""
    config = LoanDataConfig(
        n_samples=args.n_samples,
        default_rate=args.default_rate,
        random_seed=args.random_seed
    )
    
    generator = LoanDataGenerator(config)
    df = generator.generate_features()
    
    output_path = Path(args.output) if args.output else Path("data/synthetic_loans.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Generated {len(df)} loan records saved to {output_path}")
    
    # Print summary statistics
    print(f"\nData Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Default rate: {df['loan_default'].mean():.2%}")
    print(f"Average credit score: {df['credit_score'].mean():.0f}")
    print(f"Average income: ${df['annual_income'].mean():,.0f}")


def quick_train(args):
    """Quick model training and evaluation."""
    from sklearn.model_selection import train_test_split
    from features import LoanFeatureEngineer
    from data import LoanDataPreprocessor
    
    logger.info("Starting quick training...")
    
    # Generate data
    config = LoanDataConfig(n_samples=args.n_samples, random_seed=args.random_seed)
    generator = LoanDataGenerator(config)
    df = generator.generate_features()
    
    # Feature engineering
    feature_engineer = LoanFeatureEngineer()
    df_engineered = feature_engineer.engineer_all_features(df)
    
    # Preprocessing
    preprocessor = LoanDataPreprocessor()
    X, y = preprocessor.fit_transform(df_engineered)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    if args.model == 'random_forest':
        model = RandomForestModel(n_estimators=50, random_state=42)
    elif args.model == 'xgboost':
        model = XGBoostModel(n_estimators=50, random_state=42)
    elif args.model == 'lightgbm':
        model = LightGBMModel(n_estimators=50, random_state=42)
    elif args.model == 'logistic':
        model = LogisticRegressionModel(random_state=42)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(y_test, y_pred, y_prob, args.model)
    
    print(f"\n{args.model.upper()} Performance:")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Gini: {metrics['gini']:.4f}")
    print(f"KS: {metrics['ks_statistic']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Loan Default Prediction Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate data command
    data_parser = subparsers.add_parser('generate-data', help='Generate synthetic loan data')
    data_parser.add_argument('--n-samples', type=int, default=10000, help='Number of samples')
    data_parser.add_argument('--default-rate', type=float, default=0.15, help='Default rate')
    data_parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    data_parser.add_argument('--output', type=str, help='Output file path')
    
    # Quick train command
    train_parser = subparsers.add_parser('quick-train', help='Quick model training')
    train_parser.add_argument('--model', choices=['random_forest', 'xgboost', 'lightgbm', 'logistic'], 
                             default='random_forest', help='Model to train')
    train_parser.add_argument('--n-samples', type=int, default=5000, help='Number of samples')
    train_parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.command == 'generate-data':
        generate_data(args)
    elif args.command == 'quick-train':
        quick_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
