# Loan Default Prediction System

A comprehensive machine learning system for predicting loan defaults using various borrower characteristics and financial features.

## ⚠️ DISCLAIMER

**This is a research and educational project only. This system is NOT intended for investment advice or production use in financial services. Predictions may be inaccurate and should not be used for actual lending decisions.**

## Overview

This project implements a complete pipeline for loan default prediction, including:

- **Data Generation**: Synthetic loan data with realistic feature relationships
- **Feature Engineering**: Comprehensive feature creation and transformation
- **Multiple Models**: Random Forest, XGBoost, LightGBM, Logistic Regression, and Ensemble
- **Credit-Specific Metrics**: AUC, Gini coefficient, KS statistic, calibration curves
- **Explainability**: SHAP-based model interpretation
- **Interactive Demo**: Streamlit web application

## Project Structure

```
loan-default-prediction/
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── data.py            # Data generation and preprocessing
│   ├── features.py        # Feature engineering
│   ├── models.py          # ML model implementations
│   ├── evaluation.py      # Evaluation metrics and visualization
│   ├── explainability.py  # SHAP explainability
│   └── config.py          # Configuration management
├── scripts/               # Training and utility scripts
│   └── train.py          # Main training pipeline
├── configs/               # Configuration files
│   └── default.yaml      # Default configuration
├── demo/                  # Streamlit demo application
│   └── app.py            # Main demo app
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
├── data/                  # Data storage
├── models/                # Trained model storage
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Loan-Default-Prediction-System.git
   cd Loan-Default-Prediction-System
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Train Models

Run the main training script:

```bash
python scripts/train.py --n_samples 10000 --random_seed 42
```

This will:
- Generate synthetic loan data
- Apply feature engineering
- Train multiple models
- Evaluate performance
- Save models and results

### 2. Launch Demo Application

Start the Streamlit demo:

```bash
streamlit run demo/app.py
```

The demo provides:
- Interactive data visualization
- Model training and evaluation
- Loan default prediction interface
- Model explainability analysis

### 3. Configuration

Modify `configs/default.yaml` to customize:
- Data generation parameters
- Model hyperparameters
- Evaluation settings

## Features

### Data Generation

- **Realistic Features**: Credit score, income, loan amount, DTI ratio, employment history
- **Feature Relationships**: Default probability based on realistic feature interactions
- **Configurable Parameters**: Sample size, default rate, feature distributions

### Feature Engineering

- **Interaction Features**: Credit-income ratios, utilization-DTI interactions
- **Risk Buckets**: Categorical encoding of continuous features
- **Temporal Features**: Employment stability, account diversity scores
- **Polynomial Features**: Non-linear feature combinations

### Models

- **Random Forest**: Ensemble of decision trees with feature importance
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: Fast gradient boosting framework
- **Logistic Regression**: Linear baseline model
- **Ensemble**: Weighted combination of tree-based models

### Evaluation Metrics

**Classification Metrics**:
- Accuracy, Precision, Recall, F1-Score

**Credit-Specific Metrics**:
- AUC-ROC (Area Under ROC Curve)
- Gini Coefficient
- Kolmogorov-Smirnov Statistic
- Brier Score
- Calibration Error

**Visualizations**:
- ROC Curves
- Precision-Recall Curves
- Calibration Curves
- Confusion Matrices
- Feature Importance Plots

### Explainability

- **SHAP Integration**: Model-agnostic explanations
- **Feature Importance**: Global and local importance
- **Individual Predictions**: Detailed explanation for single predictions
- **Model Comparison**: Compare explanations across models

## Usage Examples

### Basic Training

```python
from src.data import LoanDataGenerator, LoanDataConfig
from src.models import RandomForestModel
from src.evaluation import ModelEvaluator

# Generate data
config = LoanDataConfig(n_samples=10000)
generator = LoanDataGenerator(config)
df = generator.generate_features()

# Train model
model = RandomForestModel()
model.fit(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(y_test, y_pred, y_prob)
```

### Feature Engineering

```python
from src.features import LoanFeatureEngineer

engineer = LoanFeatureEngineer()
df_engineered = engineer.engineer_all_features(df)
```

### Model Explainability

```python
from src.explainability import SHAPExplainer

explainer = SHAPExplainer(model, feature_names)
explainer.create_explainer(X_background)
shap_values = explainer.explain_predictions(X_test)
explainer.plot_summary(X_test)
```

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
data:
  n_samples: 10000
  default_rate: 0.15
  test_size: 0.2

model:
  random_forest:
    n_estimators: 100
    max_depth: null
    random_state: 42

evaluation:
  metrics: [accuracy, auc_roc, gini, ks_statistic]
  plots:
    roc_curve: true
    calibration_curve: true
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Development

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting
- **Pre-commit**: Git hooks for quality checks

Setup pre-commit hooks:

```bash
pre-commit install
```

### Adding New Models

1. Inherit from `BaseModel` in `src/models.py`
2. Implement required methods: `fit`, `predict`, `predict_proba`
3. Add to model registry in training script

### Adding New Features

1. Add feature creation methods to `LoanFeatureEngineer`
2. Update `engineer_all_features` method
3. Test with different datasets

## Performance

Typical performance on synthetic data:
- **AUC-ROC**: 0.75-0.85
- **Gini Coefficient**: 0.50-0.70
- **KS Statistic**: 0.30-0.50

## Limitations

- **Synthetic Data**: Uses generated data, not real loan data
- **Feature Set**: Limited to basic borrower characteristics
- **Model Complexity**: Focuses on interpretable models
- **Temporal Aspects**: No time series or temporal dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

This project is for educational and research purposes only.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{loan_default_prediction,
  title={Loan Default Prediction System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Loan-Default-Prediction-System}
}
```

## Support

For questions or issues:
- Create an issue on GitHub
- Check the documentation
- Review the demo application

---

**Remember**: This is a research demonstration. Not for production use or investment advice.
# Loan-Default-Prediction-System
