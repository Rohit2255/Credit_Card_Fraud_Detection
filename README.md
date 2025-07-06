# 💳 Credit Card Fraud Detection (Advanced)

📌 **Day 2 of #100DaysOfDataScience**

This project uses **Logistic Regression**, **Random Forest**, and **XGBoost** to detect fraudulent credit card transactions.  
We also handle severe class imbalance using **SMOTE**.

---

## 🚀 Problem Statement

> Identify fraudulent transactions in anonymized credit card data using machine learning.

Fraud makes up only **0.17%** of the dataset, so accuracy alone is not enough.  
We focus on more relevant metrics like **precision**, **recall**, and **ROC-AUC** to evaluate the model.

---

## 📊 Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Rows**: 284,807  
- **Features**: 30 (V1 to V28, Amount, Time, Class)

---

## ⚙️ Techniques Used

- Exploratory Data Analysis (EDA)
- Data Preprocessing (scaling, splitting)
- **SMOTE** for class balancing
- **Logistic Regression** (baseline)
- **Random Forest** (with feature importances)
- **XGBoost** (for performance comparison)
- ROC Curve comparison

---

## 🧠 Model Performance

### ✅ Random Forest (with SMOTE):
- Accuracy: **~99.9%**
- Precision (Fraud): **~84%**
- Recall (Fraud): **~85%**
- ROC-AUC Score: **0.98+**

---

## 📈 Visualizations

- Confusion Matrix (Seaborn + Plotly)
- Feature Importances (Plotly)
- ROC Curve Comparison: Random Forest vs XGBoost

---

## 📦 Requirements

```bash
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
imbalanced-learn
