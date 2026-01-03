"""Streamlit demo application for loan default prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import LoanDataGenerator, LoanDataConfig, LoanDataPreprocessor
from features import LoanFeatureEngineer
from models import RandomForestModel, XGBoostModel, LightGBMModel, LogisticRegressionModel
from evaluation import ModelEvaluator, CreditMetrics

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏦 Loan Default Prediction System</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ DISCLAIMER:</strong> This is a research and educational demonstration only. 
    This system is NOT intended for investment advice or production use in financial services. 
    Predictions may be inaccurate and should not be used for actual lending decisions.
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
st.sidebar.subheader("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "XGBoost", "LightGBM", "Logistic Regression", "Ensemble"],
    index=0
)

# Data generation parameters
st.sidebar.subheader("Data Parameters")
n_samples = st.sidebar.slider("Number of Samples", 1000, 50000, 10000)
default_rate = st.sidebar.slider("Default Rate", 0.05, 0.30, 0.15)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Training", "📈 Predictions", "🔍 Explainability"])

with tab1:
    st.header("Data Overview")
    
    # Generate sample data
    if st.button("Generate Sample Data", key="generate_data"):
        with st.spinner("Generating data..."):
            config = LoanDataConfig(n_samples=n_samples, default_rate=default_rate)
            generator = LoanDataGenerator(config)
            df = generator.generate_features()
            
            # Store in session state
            st.session_state['loan_data'] = df
    
    if 'loan_data' in st.session_state:
        df = st.session_state['loan_data']
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Default Rate", f"{df['loan_default'].mean():.1%}")
        with col3:
            st.metric("Avg Credit Score", f"{df['credit_score'].mean():.0f}")
        with col4:
            st.metric("Avg Income", f"${df['annual_income'].mean():,.0f}")
        
        # Data distribution plots
        st.subheader("Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='credit_score', color='loan_default', 
                            title='Credit Score Distribution by Default Status',
                            nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='annual_income', color='loan_default',
                            title='Income Distribution by Default Status',
                            nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Sample Data")
        st.dataframe(df.head(100), use_container_width=True)

with tab2:
    st.header("Model Training & Evaluation")
    
    if 'loan_data' in st.session_state:
        df = st.session_state['loan_data']
        
        if st.button("Train Models", key="train_models"):
            with st.spinner("Training models..."):
                # Feature engineering
                feature_engineer = LoanFeatureEngineer()
                df_engineered = feature_engineer.engineer_all_features(df)
                
                # Preprocessing
                preprocessor = LoanDataPreprocessor()
                X, y = preprocessor.fit_transform(df_engineered)
                
                # Train-test split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train models
                models = {
                    'Random Forest': RandomForestModel(n_estimators=100, random_state=42),
                    'XGBoost': XGBoostModel(n_estimators=100, random_state=42),
                    'LightGBM': LightGBMModel(n_estimators=100, random_state=42),
                    'Logistic Regression': LogisticRegressionModel(random_state=42)
                }
                
                # Train each model
                trained_models = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    trained_models[name] = model
                
                # Store in session state
                st.session_state['trained_models'] = trained_models
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = df_engineered.columns.tolist()
                
                st.success("Models trained successfully!")
    
    if 'trained_models' in st.session_state:
        trained_models = st.session_state['trained_models']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Model comparison
        st.subheader("Model Performance Comparison")
        
        evaluator = ModelEvaluator()
        results = []
        
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            metrics = evaluator.evaluate_model(y_test, y_pred, y_prob, name)
            results.append({
                'Model': name,
                'AUC-ROC': metrics['auc_roc'],
                'Gini': metrics['gini'],
                'KS': metrics['ks_statistic'],
                'Accuracy': metrics['accuracy'],
                'F1-Score': metrics['f1_score']
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df, x='Model', y='AUC-ROC', 
                        title='AUC-ROC by Model')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(results_df, x='Model', y='Gini', 
                        title='Gini Coefficient by Model')
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Loan Default Prediction")
    
    if 'trained_models' in st.session_state:
        trained_models = st.session_state['trained_models']
        
        st.subheader("Enter Borrower Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            annual_income = st.number_input("Annual Income ($)", 20000, 200000, 60000)
            loan_amount = st.number_input("Loan Amount ($)", 5000, 100000, 25000)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.05, 0.50, 0.25)
        
        with col2:
            previous_default = st.selectbox("Previous Default", ["No", "Yes"])
            employment_length = st.slider("Employment Length (years)", 0, 30, 5)
            num_accounts = st.slider("Number of Accounts", 1, 20, 8)
            delinquencies = st.slider("Delinquencies (2 years)", 0, 10, 0)
            revolving_util = st.slider("Revolving Utilization", 0.0, 1.0, 0.3)
        
        # Create input data
        input_data = {
            'credit_score': credit_score,
            'annual_income': annual_income,
            'loan_amount': loan_amount,
            'debt_to_income_ratio': debt_to_income,
            'previous_default': 1 if previous_default == "Yes" else 0,
            'employment_length_years': employment_length,
            'number_of_accounts': num_accounts,
            'delinquencies_2yrs': delinquencies,
            'revolving_utilization': revolving_util,
            'loan_purpose': 'debt_consolidation',
            'home_ownership': 'rent'
        }
        
        if st.button("Predict Default Risk"):
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Feature engineering
            feature_engineer = LoanFeatureEngineer()
            input_engineered = feature_engineer.engineer_all_features(input_df)
            
            # Preprocessing
            preprocessor = LoanDataPreprocessor()
            X_input, _ = preprocessor.fit_transform(input_engineered)
            
            # Get predictions from selected model
            model = trained_models[selected_model]
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "Default" if prediction == 1 else "No Default")
            
            with col2:
                st.metric("Default Probability", f"{probability[1]:.1%}")
            
            with col3:
                st.metric("No Default Probability", f"{probability[0]:.1%}")
            
            # Risk assessment
            risk_level = "Low" if probability[1] < 0.2 else "Medium" if probability[1] < 0.5 else "High"
            st.subheader(f"Risk Assessment: {risk_level}")
            
            # Probability visualization
            fig = go.Figure(data=[
                go.Bar(x=['No Default', 'Default'], 
                      y=[probability[0], probability[1]],
                      marker_color=['green', 'red'])
            ])
            fig.update_layout(title="Prediction Probabilities",
                            yaxis_title="Probability",
                            height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please train models first in the 'Model Training' tab.")

with tab4:
    st.header("Model Explainability")
    
    if 'trained_models' in st.session_state and 'X_test' in st.session_state:
        trained_models = st.session_state['trained_models']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        feature_names = st.session_state['feature_names']
        
        model = trained_models[selected_model]
        
        # Feature importance
        st.subheader("Feature Importance")
        
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)
            
            # Plot
            fig = px.bar(importance_df.tail(20), 
                        x='Importance', y='Feature',
                        orientation='h',
                        title=f'Top 20 Feature Importance - {selected_model}')
            st.plotly_chart(fig, use_container_width=True)
            
            # Importance table
            st.dataframe(importance_df.tail(20), use_container_width=True)
        
        # Sample predictions explanation
        st.subheader("Sample Prediction Explanations")
        
        sample_idx = st.slider("Select Sample Index", 0, min(100, len(X_test)-1), 0)
        
        # Get prediction for sample
        sample_pred = model.predict(X_test[sample_idx:sample_idx+1])[0]
        sample_prob = model.predict_proba(X_test[sample_idx:sample_idx+1])[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sample Prediction", "Default" if sample_pred == 1 else "No Default")
            st.metric("Actual Label", "Default" if y_test[sample_idx] == 1 else "No Default")
        
        with col2:
            st.metric("Default Probability", f"{sample_prob[1]:.1%}")
            st.metric("Prediction Correct", "Yes" if sample_pred == y_test[sample_idx] else "No")
        
        # Feature values for sample
        st.subheader("Feature Values for Selected Sample")
        sample_features = pd.DataFrame({
            'Feature': feature_names,
            'Value': X_test[sample_idx]
        })
        st.dataframe(sample_features, use_container_width=True)
    
    else:
        st.info("Please train models first in the 'Model Training' tab.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Loan Default Prediction System - Research & Educational Use Only</p>
    <p>⚠️ Not for investment advice or production use</p>
</div>
""", unsafe_allow_html=True)
