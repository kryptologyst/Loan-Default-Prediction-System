"""Feature engineering utilities for loan default prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LoanFeatureEngineer:
    """Feature engineering for loan default prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_names = []
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables.
        
        Args:
            df: DataFrame with loan features.
            
        Returns:
            DataFrame with additional interaction features.
        """
        df_features = df.copy()
        
        # Credit score and income interaction
        df_features['credit_income_ratio'] = (
            df_features['credit_score'] / (df_features['annual_income'] / 1000)
        )
        
        # Loan amount relative to income
        df_features['loan_to_income_ratio'] = (
            df_features['loan_amount'] / df_features['annual_income']
        )
        
        # Credit utilization and DTI interaction
        df_features['utilization_dti_interaction'] = (
            df_features['revolving_utilization'] * df_features['debt_to_income_ratio']
        )
        
        # Employment stability score
        df_features['employment_stability'] = (
            df_features['employment_length_years'] / (df_features['delinquencies_2yrs'] + 1)
        )
        
        # Account diversity score
        df_features['account_diversity'] = (
            df_features['number_of_accounts'] / (df_features['delinquencies_2yrs'] + 1)
        )
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} interaction features")
        
        return df_features
    
    def create_risk_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk bucket features for categorical encoding.
        
        Args:
            df: DataFrame with loan features.
            
        Returns:
            DataFrame with additional risk bucket features.
        """
        df_buckets = df.copy()
        
        # Credit score buckets
        df_buckets['credit_score_bucket'] = pd.cut(
            df_buckets['credit_score'], 
            bins=[0, 580, 670, 740, 850], 
            labels=['poor', 'fair', 'good', 'excellent']
        )
        
        # Income buckets
        df_buckets['income_bucket'] = pd.cut(
            df_buckets['annual_income'], 
            bins=[0, 30000, 50000, 75000, 100000, np.inf], 
            labels=['low', 'medium_low', 'medium', 'high', 'very_high']
        )
        
        # DTI buckets
        df_buckets['dti_bucket'] = pd.cut(
            df_buckets['debt_to_income_ratio'], 
            bins=[0, 0.2, 0.36, 0.5, 1.0], 
            labels=['low', 'moderate', 'high', 'very_high']
        )
        
        # Loan amount buckets
        df_buckets['loan_amount_bucket'] = pd.cut(
            df_buckets['loan_amount'], 
            bins=[0, 10000, 25000, 50000, np.inf], 
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        logger.info("Created risk bucket features")
        
        return df_buckets
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal and seasonal features.
        
        Args:
            df: DataFrame with loan features.
            
        Returns:
            DataFrame with additional temporal features.
        """
        df_temporal = df.copy()
        
        # Employment length categories
        df_temporal['employment_category'] = pd.cut(
            df_temporal['employment_length_years'], 
            bins=[0, 1, 3, 5, 10, np.inf], 
            labels=['new', 'short', 'medium', 'long', 'very_long']
        )
        
        # Account age proxy (based on number of accounts)
        df_temporal['account_age_proxy'] = np.log1p(df_temporal['number_of_accounts'])
        
        # Risk score based on multiple factors
        df_temporal['composite_risk_score'] = (
            (1 - df_temporal['credit_score'] / 850) * 0.3 +
            df_temporal['debt_to_income_ratio'] * 0.25 +
            df_temporal['revolving_utilization'] * 0.2 +
            (df_temporal['delinquencies_2yrs'] / 5) * 0.15 +
            (1 - np.clip(df_temporal['employment_length_years'] / 10, 0, 1)) * 0.1
        )
        
        logger.info("Created temporal and composite risk features")
        
        return df_temporal
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for important variables.
        
        Args:
            df: DataFrame with loan features.
            degree: Degree of polynomial features.
            
        Returns:
            DataFrame with additional polynomial features.
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        df_poly = df.copy()
        
        # Select important numerical features for polynomial expansion
        important_features = [
            'credit_score', 'debt_to_income_ratio', 'revolving_utilization'
        ]
        
        # Ensure features exist
        available_features = [f for f in important_features if f in df_poly.columns]
        
        if available_features:
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            poly_features = poly.fit_transform(df_poly[available_features])
            
            # Create feature names
            feature_names = poly.get_feature_names_out(available_features)
            
            # Add polynomial features to dataframe
            for i, name in enumerate(feature_names):
                if name not in df_poly.columns:  # Avoid duplicates
                    df_poly[f'poly_{name}'] = poly_features[:, i]
            
            logger.info(f"Created {len(feature_names)} polynomial features")
        
        return df_poly
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps.
        
        Args:
            df: DataFrame with loan features.
            
        Returns:
            DataFrame with all engineered features.
        """
        logger.info("Starting comprehensive feature engineering")
        
        # Apply all feature engineering steps
        df_engineered = self.create_interaction_features(df)
        df_engineered = self.create_risk_buckets(df_engineered)
        df_engineered = self.create_temporal_features(df_engineered)
        df_engineered = self.create_polynomial_features(df_engineered, degree=2)
        
        # Store feature names
        self.feature_names = df_engineered.columns.tolist()
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        
        return df_engineered
