# 💳 Online Transaction Fraud Detection System

A Machine Learning based Web Application that detects fraudulent credit card transactions using a trained Random Forest model.

---

## 🚀 Project Overview

This project uses Machine Learning to classify online credit card transactions as:

- ✅ Genuine
- 🚨 Fraudulent

The system is built using:

- Python
- Scikit-learn
- Random Forest Classifier
- Flask (Web Framework)
- HTML/CSS (Frontend)

---

## 📊 Dataset

The dataset contains:

- Time
- V1 to V28 (PCA transformed features)
- Amount
- Class (0 = Genuine, 1 = Fraud)

The V1–V28 features are anonymized PCA components to protect user identity.

---

## 🧠 Machine Learning Model

Model Used:
- Random Forest Classifier

Steps:
1. Data Preprocessing
2. Feature Scaling
3. Train-Test Split
4. Model Training
5. Model Evaluation
6. Model Saving (.pkl)
7. Web Deployment using Flask

---

## 🌐 How It Works

1. User enters 30 transaction values.
2. Flask receives input.
3. Data is converted to NumPy array.
4. Model predicts:
   - Class (Fraud or Genuine)
   - Probability score
5. Result is displayed with animated gauge UI.

---

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Flask
- HTML/CSS
- JavaScript
- Git & GitHub

---

## 📁 Project Structure

```
Online-Transaction-Fraud-Detection/
│
├── app.py
├── fraud_detection.ipynb
├── fraud_detection_model.pkl
├── scaler.pkl
├── templates/
│   └── index.html
└── README.md
```

---

## 🎯 Future Improvements

- Real-time transaction monitoring
- Database integration
- Cloud deployment
- REST API integration
- Dashboard analytics

---

## 👨‍💻 Author

Mayank Ambade  
Machine Learning & Web Development Project
