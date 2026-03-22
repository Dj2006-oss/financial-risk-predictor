# 💳 Financial Risk Prediction System

## 📌 Overview
This project is a machine learning-based system designed to detect fraudulent financial transactions using a real-world dataset. The model predicts whether a transaction is **safe or fraudulent**.

---

## 🎯 Objective
To build a system that can accurately detect fraud while handling the challenge of **imbalanced data**, where fraudulent cases are very rare.

---

## 📊 Dataset
- Source: Kaggle - Credit Card Fraud Detection Dataset  
- Total transactions: 284,807  
- Features: 30  
- Target:
  - 0 → Normal transaction  
  - 1 → Fraudulent transaction  

---

## 🔗 Dataset Link
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  

---

## 🧠 Approach

### 1. Data Analysis
- Explored dataset using Pandas  
- Checked class distribution and missing values  

### 2. Data Preprocessing
- Separated features (X) and target (y)  
- Split data into training and testing sets (80-20 split)  

### 3. Model Building
- Used Logistic Regression  
- Applied **class weighting** to handle imbalance:
  ```python
  class_weight='balanced'
