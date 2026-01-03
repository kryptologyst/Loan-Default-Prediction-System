"""Risk management and governance utilities for loan default prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import warnings

logger = logging.getLogger(__name__)


class DataLeakageDetector:
    """Detect and prevent data leakage in loan default prediction."""
    
    def __init__(self):
        """Initialize the leakage detector."""
        self.leakage_checks = []
        
    def check_feature_target_correlation(self, X: pd.DataFrame, y: pd.Series, 
                                       threshold: float = 0.95) -> List[str]:
        """Check for suspiciously high correlations between features and target.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            threshold: Correlation threshold for flagging.
            
        Returns:
            List of suspicious features.
        """
        suspicious_features = []
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                continue
                
            corr = abs(X[col].corr(y))
            if corr > threshold:
                suspicious_features.append(col)
                self.leakage_checks.append({
                    'check': 'feature_target_correlation',
                    'feature': col,
                    'correlation': corr,
                    'threshold': threshold,
                    'status': 'FAILED'
                })
                logger.warning(f"Suspicious correlation between {col} and target: {corr:.3f}")
        
        return suspicious_features
    
    def check_future_information(self, df: pd.DataFrame, 
                                date_col: str = 'application_date',
                                target_col: str = 'loan_default') -> List[str]:
        """Check for future information leakage.
        
        Args:
            df: DataFrame with temporal information.
            date_col: Name of date column.
            target_col: Name of target column.
            
        Returns:
            List of potential leakage issues.
        """
        issues = []
        
        if date_col not in df.columns:
            logger.warning(f"Date column {date_col} not found. Skipping temporal checks.")
            return issues
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Check if any features are calculated after application date
        for col in df.columns:
            if col in [date_col, target_col]:
                continue
                
            # This is a simplified check - in practice, you'd need domain knowledge
            # about when each feature becomes available
            if 'future' in col.lower() or 'post' in col.lower():
                issues.append(f"Feature {col} may contain future information")
                self.leakage_checks.append({
                    'check': 'future_information',
                    'feature': col,
                    'issue': 'Potential future information',
                    'status': 'WARNING'
                })
        
        return issues
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality issues that could indicate leakage.
        
        Args:
            df: DataFrame to check.
            
        Returns:
            Dictionary of quality issues.
        """
        issues = {
            'missing_values': {},
            'duplicates': 0,
            'constant_features': [],
            'high_cardinality': []
        }
        
        # Missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        issues['missing_values'] = missing_pct[missing_pct > 0].to_dict()
        
        # Duplicates
        issues['duplicates'] = df.duplicated().sum()
        
        # Constant features
        for col in df.columns:
            if df[col].nunique() <= 1:
                issues['constant_features'].append(col)
        
        # High cardinality categorical features
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.5:
                issues['high_cardinality'].append(col)
        
        return issues
    
    def generate_report(self) -> str:
        """Generate leakage detection report.
        
        Returns:
            Formatted report string.
        """
        report = "=== DATA LEAKAGE DETECTION REPORT ===\n\n"
        
        if not self.leakage_checks:
            report += "No leakage issues detected.\n"
            return report
        
        failed_checks = [check for check in self.leakage_checks if check['status'] == 'FAILED']
        warning_checks = [check for check in self.leakage_checks if check['status'] == 'WARNING']
        
        if failed_checks:
            report += "FAILED CHECKS:\n"
            for check in failed_checks:
                report += f"- {check['check']}: {check.get('feature', 'N/A')}\n"
        
        if warning_checks:
            report += "\nWARNINGS:\n"
            for check in warning_checks:
                report += f"- {check['check']}: {check.get('feature', 'N/A')}\n"
        
        return report


class TimeBasedSplitter:
    """Time-based data splitting for loan default prediction."""
    
    def __init__(self, date_col: str = 'application_date', 
                 test_size: float = 0.2, validation_size: float = 0.2):
        """Initialize time-based splitter.
        
        Args:
            date_col: Name of date column.
            test_size: Proportion of data for testing.
            validation_size: Proportion of data for validation.
        """
        self.date_col = date_col
        self.test_size = test_size
        self.validation_size = validation_size
        
    def split(self, df: pd.DataFrame, target_col: str = 'loan_default') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data based on time.
        
        Args:
            df: DataFrame with temporal data.
            target_col: Name of target column.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if self.date_col not in df.columns:
            logger.warning(f"Date column {self.date_col} not found. Using random split.")
            from sklearn.model_selection import train_test_split
            train_df, temp_df = train_test_split(df, test_size=self.test_size + self.validation_size, 
                                               random_state=42, stratify=df[target_col])
            val_df, test_df = train_test_split(temp_df, test_size=self.test_size/(self.test_size + self.validation_size),
                                             random_state=42, stratify=temp_df[target_col])
            return train_df, val_df, test_df
        
        # Convert to datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Sort by date
        df_sorted = df.sort_values(self.date_col)
        
        # Calculate split points
        n_total = len(df_sorted)
        n_test = int(n_total * self.test_size)
        n_val = int(n_total * self.validation_size)
        n_train = n_total - n_test - n_val
        
        # Split by time
        train_df = df_sorted.iloc[:n_train].copy()
        val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
        test_df = df_sorted.iloc[n_train + n_val:].copy()
        
        logger.info(f"Time-based split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df


class PurgedCrossValidation:
    """Purged cross-validation for overlapping samples."""
    
    def __init__(self, n_splits: int = 5, purge_days: int = 30, embargo_days: int = 30):
        """Initialize purged CV.
        
        Args:
            n_splits: Number of CV splits.
            purge_days: Days to purge around each split.
            embargo_days: Days to embargo after each split.
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        
    def split(self, X: pd.DataFrame, y: pd.Series, 
              date_col: str = 'application_date') -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged CV splits.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            date_col: Name of date column.
            
        Returns:
            List of (train_indices, test_indices) tuples.
        """
        if date_col not in X.columns:
            logger.warning(f"Date column {date_col} not found. Using standard CV.")
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            return list(kf.split(X))
        
        # Convert to datetime
        X[date_col] = pd.to_datetime(X[date_col])
        
        # Sort by date
        sorted_indices = X[date_col].argsort()
        X_sorted = X.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]
        
        splits = []
        n_samples = len(X_sorted)
        
        for i in range(self.n_splits):
            # Calculate test set boundaries
            test_start = int(i * n_samples / self.n_splits)
            test_end = int((i + 1) * n_samples / self.n_splits)
            
            # Get test set dates
            test_dates = X_sorted[date_col].iloc[test_start:test_end]
            test_start_date = test_dates.min()
            test_end_date = test_dates.max()
            
            # Calculate purge and embargo periods
            purge_start = test_start_date - timedelta(days=self.purge_days)
            purge_end = test_end_date + timedelta(days=self.purge_days)
            embargo_start = test_end_date
            embargo_end = test_end_date + timedelta(days=self.embargo_days)
            
            # Create train mask (exclude purge and embargo periods)
            train_mask = (
                (X_sorted[date_col] < purge_start) | 
                (X_sorted[date_col] > embargo_end)
            )
            
            train_indices = sorted_indices[train_mask]
            test_indices = sorted_indices[test_start:test_end]
            
            splits.append((train_indices, test_indices))
            
            logger.info(f"Split {i+1}: Train={len(train_indices)}, Test={len(test_indices)}")
        
        return splits


class ModelGovernance:
    """Model governance and monitoring utilities."""
    
    def __init__(self):
        """Initialize model governance."""
        self.model_registry = {}
        self.performance_history = []
        
    def register_model(self, model_name: str, model: Any, 
                      metadata: Dict[str, Any]) -> None:
        """Register a model in the governance system.
        
        Args:
            model_name: Name of the model.
            model: Model object.
            metadata: Model metadata.
        """
        self.model_registry[model_name] = {
            'model': model,
            'metadata': metadata,
            'created_at': datetime.now(),
            'status': 'active'
        }
        
        logger.info(f"Registered model: {model_name}")
    
    def log_performance(self, model_name: str, metrics: Dict[str, float],
                       dataset_name: str = 'test') -> None:
        """Log model performance metrics.
        
        Args:
            model_name: Name of the model.
            metrics: Performance metrics.
            dataset_name: Name of the dataset.
        """
        performance_record = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'timestamp': datetime.now(),
            'metrics': metrics
        }
        
        self.performance_history.append(performance_record)
        logger.info(f"Logged performance for {model_name} on {dataset_name}")
    
    def check_model_drift(self, model_name: str, 
                         current_metrics: Dict[str, float],
                         threshold: float = 0.05) -> Dict[str, Any]:
        """Check for model performance drift.
        
        Args:
            model_name: Name of the model.
            current_metrics: Current performance metrics.
            threshold: Performance degradation threshold.
            
        Returns:
            Dictionary with drift analysis.
        """
        # Get historical performance for this model
        historical_performance = [
            record for record in self.performance_history 
            if record['model_name'] == model_name
        ]
        
        if len(historical_performance) < 2:
            return {'status': 'insufficient_data', 'message': 'Not enough historical data'}
        
        # Calculate baseline performance
        baseline_metrics = {}
        for metric_name in current_metrics.keys():
            historical_values = [
                record['metrics'].get(metric_name) 
                for record in historical_performance
                if record['metrics'].get(metric_name) is not None
            ]
            
            if historical_values:
                baseline_metrics[metric_name] = np.mean(historical_values)
        
        # Check for drift
        drift_detected = {}
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                degradation = (baseline_value - current_value) / baseline_value
                
                if degradation > threshold:
                    drift_detected[metric_name] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation': degradation
                    }
        
        return {
            'status': 'drift_detected' if drift_detected else 'no_drift',
            'drift_metrics': drift_detected,
            'baseline_metrics': baseline_metrics
        }
    
    def generate_governance_report(self) -> str:
        """Generate model governance report.
        
        Returns:
            Formatted governance report.
        """
        report = "=== MODEL GOVERNANCE REPORT ===\n\n"
        
        # Model registry
        report += f"Registered Models: {len(self.model_registry)}\n"
        for name, info in self.model_registry.items():
            report += f"- {name}: {info['status']} (created: {info['created_at']})\n"
        
        # Performance history
        report += f"\nPerformance Records: {len(self.performance_history)}\n"
        
        # Recent performance
        if self.performance_history:
            recent_performance = self.performance_history[-5:]
            report += "\nRecent Performance:\n"
            for record in recent_performance:
                report += f"- {record['model_name']} ({record['dataset_name']}): {record['timestamp']}\n"
        
        return report


class RiskManager:
    """Risk management utilities for loan default prediction."""
    
    def __init__(self):
        """Initialize risk manager."""
        self.risk_limits = {}
        self.risk_alerts = []
        
    def set_risk_limits(self, limits: Dict[str, float]) -> None:
        """Set risk limits for model monitoring.
        
        Args:
            limits: Dictionary of risk limits.
        """
        self.risk_limits = limits
        logger.info(f"Set risk limits: {limits}")
    
    def check_risk_limits(self, metrics: Dict[str, float]) -> List[str]:
        """Check if metrics exceed risk limits.
        
        Args:
            metrics: Current metrics to check.
            
        Returns:
            List of risk limit violations.
        """
        violations = []
        
        for metric_name, value in metrics.items():
            if metric_name in self.risk_limits:
                limit = self.risk_limits[metric_name]
                
                if value > limit:
                    violation = f"{metric_name}: {value:.3f} > {limit:.3f}"
                    violations.append(violation)
                    
                    # Log alert
                    alert = {
                        'timestamp': datetime.now(),
                        'metric': metric_name,
                        'value': value,
                        'limit': limit,
                        'severity': 'HIGH' if value > limit * 1.5 else 'MEDIUM'
                    }
                    self.risk_alerts.append(alert)
                    
                    logger.warning(f"Risk limit violation: {violation}")
        
        return violations
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary.
        
        Returns:
            Dictionary with risk summary.
        """
        return {
            'risk_limits': self.risk_limits,
            'total_alerts': len(self.risk_alerts),
            'high_severity_alerts': len([a for a in self.risk_alerts if a['severity'] == 'HIGH']),
            'recent_alerts': self.risk_alerts[-5:] if self.risk_alerts else []
        }
