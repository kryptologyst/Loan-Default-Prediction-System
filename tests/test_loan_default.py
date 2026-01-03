"""Tests for loan default prediction system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import LoanDataGenerator, LoanDataConfig, LoanDataPreprocessor
from features import LoanFeatureEngineer
from models import RandomForestModel, XGBoostModel, LightGBMModel, LogisticRegressionModel
from evaluation import ModelEvaluator, CreditMetrics
from governance import DataLeakageDetector, TimeBasedSplitter, ModelGovernance


class TestLoanDataGenerator:
    """Test cases for loan data generation."""
    
    def test_data_generation(self):
        """Test basic data generation."""
        config = LoanDataConfig(n_samples=1000, random_seed=42)
        generator = LoanDataGenerator(config)
        df = generator.generate_features()
        
        assert len(df) == 1000
        assert 'loan_default' in df.columns
        assert 'credit_score' in df.columns
        assert 'annual_income' in df.columns
        
    def test_default_rate(self):
        """Test default rate configuration."""
        config = LoanDataConfig(n_samples=1000, default_rate=0.2, random_seed=42)
        generator = LoanDataGenerator(config)
        df = generator.generate_features()
        
        # Allow some tolerance for randomness
        default_rate = df['loan_default'].mean()
        assert 0.15 <= default_rate <= 0.25
    
    def test_feature_ranges(self):
        """Test feature value ranges."""
        config = LoanDataConfig(n_samples=100, random_seed=42)
        generator = LoanDataGenerator(config)
        df = generator.generate_features()
        
        assert df['credit_score'].min() >= 300
        assert df['credit_score'].max() <= 850
        assert df['annual_income'].min() > 0
        assert df['loan_amount'].min() > 0
        assert df['debt_to_income_ratio'].min() >= 0.05
        assert df['debt_to_income_ratio'].max() <= 0.45


class TestLoanDataPreprocessor:
    """Test cases for data preprocessing."""
    
    def test_preprocessing(self):
        """Test basic preprocessing."""
        # Create sample data
        data = {
            'credit_score': [700, 650, 750],
            'annual_income': [50000, 60000, 70000],
            'loan_amount': [20000, 25000, 30000],
            'debt_to_income_ratio': [0.2, 0.3, 0.25],
            'loan_purpose': ['debt_consolidation', 'credit_card', 'home_improvement'],
            'loan_default': [0, 1, 0]
        }
        df = pd.DataFrame(data)
        
        preprocessor = LoanDataPreprocessor()
        X, y = preprocessor.fit_transform(df)
        
        assert X.shape[0] == 3
        assert X.shape[1] == 5  # 5 features
        assert len(y) == 3
        assert y.dtype == np.int64
    
    def test_transform_new_data(self):
        """Test transforming new data."""
        # Create training data
        train_data = {
            'credit_score': [700, 650],
            'annual_income': [50000, 60000],
            'loan_purpose': ['debt_consolidation', 'credit_card'],
            'loan_default': [0, 1]
        }
        train_df = pd.DataFrame(train_data)
        
        preprocessor = LoanDataPreprocessor()
        preprocessor.fit_transform(train_df)
        
        # Create test data
        test_data = {
            'credit_score': [750],
            'annual_income': [70000],
            'loan_purpose': ['debt_consolidation'],
            'loan_default': [0]
        }
        test_df = pd.DataFrame(test_data)
        
        X_test = preprocessor.transform(test_df)
        assert X_test.shape[0] == 1
        assert X_test.shape[1] == 2  # 2 features after encoding


class TestLoanFeatureEngineer:
    """Test cases for feature engineering."""
    
    def test_feature_engineering(self):
        """Test comprehensive feature engineering."""
        # Create sample data
        data = {
            'credit_score': [700, 650, 750],
            'annual_income': [50000, 60000, 70000],
            'loan_amount': [20000, 25000, 30000],
            'debt_to_income_ratio': [0.2, 0.3, 0.25],
            'revolving_utilization': [0.3, 0.5, 0.4],
            'employment_length_years': [5, 3, 7],
            'delinquencies_2yrs': [0, 1, 0],
            'loan_default': [0, 1, 0]
        }
        df = pd.DataFrame(data)
        
        engineer = LoanFeatureEngineer()
        df_engineered = engineer.engineer_all_features(df)
        
        # Should have more features than original
        assert len(df_engineered.columns) > len(df.columns)
        
        # Check for specific engineered features
        assert 'credit_income_ratio' in df_engineered.columns
        assert 'loan_to_income_ratio' in df_engineered.columns
        assert 'utilization_dti_interaction' in df_engineered.columns


class TestModels:
    """Test cases for machine learning models."""
    
    def test_random_forest(self):
        """Test Random Forest model."""
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == 100
        assert probabilities.shape == (100, 2)
        assert model.is_fitted == True
    
    def test_xgboost(self):
        """Test XGBoost model."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = XGBoostModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == 100
        assert probabilities.shape == (100, 2)
        assert model.is_fitted == True
    
    def test_lightgbm(self):
        """Test LightGBM model."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = LightGBMModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == 100
        assert probabilities.shape == (100, 2)
        assert model.is_fitted == True
    
    def test_logistic_regression(self):
        """Test Logistic Regression model."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == 100
        assert probabilities.shape == (100, 2)
        assert model.is_fitted == True


class TestEvaluation:
    """Test cases for evaluation metrics."""
    
    def test_credit_metrics(self):
        """Test credit-specific metrics."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_score = np.array([0.1, 0.8, 0.2, 0.9, 0.3])
        
        gini = CreditMetrics.gini_coefficient(y_true, y_score)
        ks = CreditMetrics.ks_statistic(y_true, y_score)
        
        assert 0 <= gini <= 1
        assert 0 <= ks <= 1
    
    def test_model_evaluator(self):
        """Test model evaluator."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.3])
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(y_true, y_pred, y_prob, "TestModel")
        
        assert 'accuracy' in metrics
        assert 'auc_roc' in metrics
        assert 'gini' in metrics
        assert 'ks_statistic' in metrics


class TestGovernance:
    """Test cases for governance utilities."""
    
    def test_data_leakage_detector(self):
        """Test data leakage detection."""
        detector = DataLeakageDetector()
        
        # Create data with suspicious correlation
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0, 1, 0, 1, 0],
            'suspicious': [0, 1, 0, 1, 0]  # Perfect correlation with target
        })
        y = pd.Series([0, 1, 0, 1, 0])
        
        suspicious_features = detector.check_feature_target_correlation(X, y, threshold=0.9)
        
        assert 'suspicious' in suspicious_features
    
    def test_time_based_splitter(self):
        """Test time-based splitting."""
        # Create data with dates
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = {
            'application_date': dates,
            'feature1': np.random.randn(100),
            'loan_default': np.random.randint(0, 2, 100)
        }
        df = pd.DataFrame(data)
        
        splitter = TimeBasedSplitter()
        train_df, val_df, test_df = splitter.split(df)
        
        assert len(train_df) + len(val_df) + len(test_df) == 100
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
    
    def test_model_governance(self):
        """Test model governance."""
        governance = ModelGovernance()
        
        # Mock model
        mock_model = Mock()
        metadata = {'version': '1.0', 'algorithm': 'RandomForest'}
        
        governance.register_model('test_model', mock_model, metadata)
        
        assert 'test_model' in governance.model_registry
        
        # Log performance
        metrics = {'accuracy': 0.85, 'auc_roc': 0.80}
        governance.log_performance('test_model', metrics)
        
        assert len(governance.performance_history) == 1


if __name__ == "__main__":
    pytest.main([__file__])
