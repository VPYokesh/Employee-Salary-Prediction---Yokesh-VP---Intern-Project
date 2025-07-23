# 💼 Employee Salary Prediction using Machine Learning

This project predicts whether an employee earns more than \$50K per year based on demographic and employment-related attributes. It uses various machine learning models and is deployed as an interactive web application using Streamlit.

## 📌 Problem Statement

The objective of this project is to classify individuals into two income categories (<=50K or >50K annually) using the UCI Adult Income dataset. The system is designed to:
- Analyze socio-economic and professional features,
- Predict income class using supervised machine learning algorithms,
- Provide a deployable and interactive prediction interface for real-world usage.

## ⚙️ Technologies Used

- **Language:** Python
- **Libraries:**
  - `pandas`, `numpy`, `matplotlib`, `seaborn` – Data handling & visualization
  - `sklearn` – Machine learning models and preprocessing
  - `xgboost` – Gradient Boosted Decision Trees
  - `tensorflow.keras` – Neural Network model
  - `joblib` – Model serialization
  - `streamlit` – Web app interface

## 🛠️ System Requirements

- Python 3.11+
- Jupyter Notebook or Google Colab
- Streamlit (for deployment)

---

## 🔄 Project Workflow

### 1. Data Preprocessing
- Removed rows with missing values or placeholders (`"?"`)
- Label encoded all categorical variables
- Detected and handled outliers
- Applied `StandardScaler` to normalize numerical features

### 2. Model Training
Trained and evaluated multiple machine learning models:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- XGBoost
- Neural Network using Keras

### 3. Model Evaluation
- Accuracy score
- Classification report (Precision, Recall, F1-Score)
- Confusion Matrix

### 4. Deployment
- Best model, encoders, and scaler saved using `joblib`
- Streamlit app built to accept user inputs and predict salary class

---

## 🚀 How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/YokeshVP/employee-salary-prediction.git
   cd employee-salary-prediction
