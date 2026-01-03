"""Configuration management for loan default prediction project."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration parameters."""
    n_samples: int = 10000
    random_seed: int = 42
    default_rate: float = 0.15
    test_size: float = 0.2
    validation_size: float = 0.2


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    random_forest: Dict[str, Any] = None
    xgboost: Dict[str, Any] = None
    lightgbm: Dict[str, Any] = None
    logistic_regression: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.random_forest is None:
            self.random_forest = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        
        if self.xgboost is None:
            self.xgboost = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        if self.lightgbm is None:
            self.lightgbm = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        if self.logistic_regression is None:
            self.logistic_regression = {
                'random_state': 42,
                'max_iter': 1000,
                'C': 1.0
            }


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    metrics: list = None
    plots: Dict[str, bool] = None
    save_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'auc_roc', 'auc_pr', 'gini', 'ks_statistic',
                'brier_score', 'calibration_error'
            ]
        
        if self.plots is None:
            self.plots = {
                'roc_curve': True,
                'precision_recall_curve': True,
                'calibration_curve': True,
                'confusion_matrix': True,
                'feature_importance': True
            }


@dataclass
class ProjectConfig:
    """Main project configuration."""
    data: DataConfig = None
    model: ModelConfig = None
    evaluation: EvaluationConfig = None
    project_name: str = "Loan Default Prediction"
    version: str = "1.0.0"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()


class ConfigManager:
    """Configuration manager for loading and saving configurations."""
    
    @staticmethod
    def load_config(config_path: str) -> ProjectConfig:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            ProjectConfig object.
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ConfigManager._dict_to_config(config_dict)
    
    @staticmethod
    def save_config(config: ProjectConfig, config_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: ProjectConfig object to save.
            config_path: Path to save configuration.
        """
        config_dict = ConfigManager._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ProjectConfig:
        """Convert dictionary to ProjectConfig object."""
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        return ProjectConfig(
            data=data_config,
            model=model_config,
            evaluation=evaluation_config,
            project_name=config_dict.get('project_name', 'Loan Default Prediction'),
            version=config_dict.get('version', '1.0.0')
        )
    
    @staticmethod
    def _config_to_dict(config: ProjectConfig) -> Dict[str, Any]:
        """Convert ProjectConfig object to dictionary."""
        return {
            'project_name': config.project_name,
            'version': config.version,
            'data': {
                'n_samples': config.data.n_samples,
                'random_seed': config.data.random_seed,
                'default_rate': config.data.default_rate,
                'test_size': config.data.test_size,
                'validation_size': config.data.validation_size
            },
            'model': {
                'random_forest': config.model.random_forest,
                'xgboost': config.model.xgboost,
                'lightgbm': config.model.lightgbm,
                'logistic_regression': config.model.logistic_regression
            },
            'evaluation': {
                'metrics': config.evaluation.metrics,
                'plots': config.evaluation.plots,
                'save_plots': config.evaluation.save_plots,
                'plot_format': config.evaluation.plot_format,
                'plot_dpi': config.evaluation.plot_dpi
            }
        }
