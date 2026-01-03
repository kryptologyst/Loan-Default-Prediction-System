"""Data generation and preprocessing utilities for loan default prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoanDataConfig:
    """Configuration for synthetic loan data generation."""
    
    n_samples: int = 10000
    random_seed: int = 42
    default_rate: float = 0.15  # 15% default rate
    credit_score_range: Tuple[int, int] = (300, 850)
    income_mean: float = 60000
    income_std: float = 15000
    loan_amount_mean: float = 25000
    loan_amount_std: float = 7000
    dti_min: float = 0.05
    dti_max: float = 0.45


class LoanDataGenerator:
    """Generate synthetic loan data for default prediction modeling."""
    
    def __init__(self, config: LoanDataConfig):
        """Initialize the data generator.
        
        Args:
            config: Configuration object for data generation parameters.
        """
        self.config = config
        np.random.seed(config.random_seed)
        
    def generate_features(self) -> pd.DataFrame:
        """Generate synthetic loan features.
        
        Returns:
            DataFrame with loan features including credit score, income, 
            loan amount, debt-to-income ratio, and previous default history.
        """
        n_samples = self.config.n_samples
        
        # Generate base features
        data = {
            'credit_score': np.random.randint(
                self.config.credit_score_range[0], 
                self.config.credit_score_range[1], 
                n_samples
            ),
            'annual_income': np.random.normal(
                self.config.income_mean, 
                self.config.income_std, 
                n_samples
            ),
            'loan_amount': np.random.normal(
                self.config.loan_amount_mean, 
                self.config.loan_amount_std, 
                n_samples
            ),
            'debt_to_income_ratio': np.random.uniform(
                self.config.dti_min, 
                self.config.dti_max, 
                n_samples
            ),
            'previous_default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'employment_length_years': np.random.exponential(5, n_samples),
            'number_of_accounts': np.random.poisson(8, n_samples),
            'delinquencies_2yrs': np.random.poisson(0.5, n_samples),
            'revolving_utilization': np.random.uniform(0, 1, n_samples),
            'loan_purpose': np.random.choice(
                ['debt_consolidation', 'credit_card', 'home_improvement', 'other'], 
                n_samples
            ),
            'home_ownership': np.random.choice(
                ['rent', 'own', 'mortgage'], 
                n_samples, 
                p=[0.4, 0.2, 0.4]
            )
        }
        
        df = pd.DataFrame(data)
        
        # Ensure positive values for income and loan amount
        df['annual_income'] = np.abs(df['annual_income'])
        df['loan_amount'] = np.abs(df['loan_amount'])
        df['employment_length_years'] = np.abs(df['employment_length_years'])
        
        # Generate realistic default labels based on feature relationships
        df['loan_default'] = self._generate_realistic_defaults(df)
        
        logger.info(f"Generated {len(df)} loan records with {df['loan_default'].mean():.2%} default rate")
        
        return df
    
    def _generate_realistic_defaults(self, df: pd.DataFrame) -> np.ndarray:
        """Generate realistic default labels based on feature relationships.
        
        Args:
            df: DataFrame with loan features.
            
        Returns:
            Array of default labels (0 or 1).
        """
        # Create probability of default based on features
        default_prob = np.zeros(len(df))
        
        # Credit score effect (lower score = higher default risk)
        credit_score_norm = (df['credit_score'] - 300) / (850 - 300)
        default_prob += (1 - credit_score_norm) * 0.3
        
        # Debt-to-income ratio effect
        default_prob += df['debt_to_income_ratio'] * 0.4
        
        # Previous default effect
        default_prob += df['previous_default'] * 0.2
        
        # Employment length effect (shorter = higher risk)
        emp_length_norm = np.clip(df['employment_length_years'] / 10, 0, 1)
        default_prob += (1 - emp_length_norm) * 0.1
        
        # Delinquencies effect
        default_prob += np.clip(df['delinquencies_2yrs'] / 5, 0, 1) * 0.15
        
        # Revolving utilization effect
        default_prob += df['revolving_utilization'] * 0.1
        
        # Add some noise
        default_prob += np.random.normal(0, 0.05, len(df))
        
        # Convert to binary labels
        default_prob = np.clip(default_prob, 0, 1)
        defaults = np.random.binomial(1, default_prob)
        
        return defaults


class LoanDataPreprocessor:
    """Preprocess loan data for machine learning models."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'loan_default') -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform data.
        
        Args:
            df: DataFrame with loan data.
            target_col: Name of the target column.
            
        Returns:
            Tuple of (features, target) arrays.
        """
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        self.scaler = StandardScaler()
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Preprocessed {len(X)} samples with {len(self.feature_names)} features")
        
        return X.values, y.values
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor.
        
        Args:
            df: DataFrame with loan data.
            
        Returns:
            Transformed feature array.
        """
        if self.scaler is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = df.copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                # Handle unseen categories
                X[col] = X[col].astype(str)
                unseen_mask = ~X[col].isin(self.label_encoders[col].classes_)
                if unseen_mask.any():
                    logger.warning(f"Found unseen categories in {col}, using most frequent class")
                    X.loc[unseen_mask, col] = self.label_encoders[col].classes_[0]
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X.values
