# 🏭 Predictive Maintenance for Injection Molding Machine

A machine learning-based predictive maintenance system designed to anticipate equipment failures and optimize maintenance schedules for injection molding machines in manufacturing environments.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## 🎯 Overview

Injection molding machines are critical assets in manufacturing. Unexpected breakdowns can lead to costly downtime, production delays, and quality issues. This project implements a predictive maintenance solution that uses machine learning to:

- Predict potential machine failures before they occur
- Identify key factors contributing to equipment degradation
- Provide actionable recommendations for quality improvement
- Optimize maintenance scheduling to reduce costs

## 📁 Project Structure

```
Predictive-maintenance-for-Injection-Molding-Machine/
│
├── Data_generation.ipynb              # Synthetic data generation for injection molding
├── step2_Advanced_feature_Engineering.ipynb  # Feature engineering & preprocessing
├── step3_model.ipynb                  # Base model training & evaluation
├── 04_Optimized_Models.ipynb          # Hyperparameter tuning & model optimization
├── 05_SHAP_Explainability.ipynb       # Model interpretability with SHAP
│
├── injection_moulding_data.csv        # Raw dataset
├── processed_data_advanced.csv        # Processed features
├── train_data_advanced.csv            # Training set
├── val_data_advanced.csv              # Validation set
├── test_data_advanced.csv             # Test set
│
├── X_train_scaled_advanced.npy        # Scaled training features
├── X_val_scaled_advanced.npy          # Scaled validation features
├── X_test_scaled_advanced.npy         # Scaled test features
├── y_train_advanced.npy               # Training labels
├── y_val_advanced.npy                 # Validation labels
├── y_test_advanced.npy                # Test labels
│
├── scaler_advanced.pkl                # Fitted scaler object
├── model_comparison_advanced.csv      # Model performance comparison
├── optimization_results.csv           # Hyperparameter tuning results
├── quality_improvement_recommendations.csv  # Generated recommendations
│
└── 05_feature_importance_advanced.png # Feature importance visualization
```

## ✨ Features

- **Data Generation**: Realistic synthetic data simulating injection molding process parameters
- **Advanced Feature Engineering**: Domain-specific features capturing machine behavior patterns
- **Multiple ML Models**: Comparison of various algorithms for optimal performance
- **Hyperparameter Optimization**: Fine-tuned models for best predictive accuracy
- **Explainable AI**: SHAP-based model interpretability for actionable insights
- **Quality Recommendations**: Automated suggestions for process improvement

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/anita2210/Predictive-maintenance-for-Injection-Molding-Machine.git
cd Predictive-maintenance-for-Injection-Molding-Machine
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap jupyter
```

## 🚀 Usage

Run the notebooks in sequence:

1. **Data Generation** - Generate synthetic injection molding data
```bash
jupyter notebook Data_generation.ipynb
```

2. **Feature Engineering** - Process and engineer features
```bash
jupyter notebook step2_Advanced_feature_Engineering.ipynb
```

3. **Model Training** - Train and evaluate base models
```bash
jupyter notebook step3_model.ipynb
```

4. **Model Optimization** - Tune hyperparameters
```bash
jupyter notebook 04_Optimized_Models.ipynb
```

5. **Explainability** - Interpret model predictions
```bash
jupyter notebook 05_SHAP_Explainability.ipynb
```

## 📊 Methodology

### 1. Data Generation
- Simulated realistic injection molding process parameters
- Included features: temperature, pressure, cycle time, vibration, etc.
- Generated failure labels based on domain knowledge

### 2. Feature Engineering
- Created rolling statistics (mean, std, min, max)
- Engineered interaction features
- Applied scaling and normalization
- Handled class imbalance

### 3. Model Development
- Trained multiple classifiers (Random Forest, XGBoost, etc.)
- Performed cross-validation
- Optimized hyperparameters using grid/random search

### 4. Model Explainability
- Used SHAP values for feature importance
- Identified key drivers of machine failures
- Generated actionable maintenance recommendations

## 📈 Results

| Metric | Value |
|--------|-------|
| Model | [Best Model Name] |
| Accuracy | XX% |
| Precision | XX% |
| Recall | XX% |
| F1-Score | XX% |

### Key Findings
- Top predictive features identified through SHAP analysis
- Quality improvement recommendations generated
- Maintenance optimization potential demonstrated

## 🔧 Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning models
- **SHAP** - Model explainability
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development

## 🔮 Future Improvements

- [ ] Real-time prediction API deployment
- [ ] Integration with IoT sensors
- [ ] Dashboard for monitoring predictions
- [ ] Deep learning models (LSTM for time-series)
- [ ] Anomaly detection module
- [ ] Cost-benefit analysis for maintenance decisions

## 📧 Contact

**Anita** - [GitHub Profile](https://github.com/anita2210)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---
⭐ If you found this project helpful, please give it a star!
