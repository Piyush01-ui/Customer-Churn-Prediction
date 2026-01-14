# 🚀 AI Customer Churn Intelligence Platform

An end-to-end **Machine Learning + Explainable AI** project that predicts customer churn for **SaaS & subscription businesses**, explains *why* customers churn, and provides actionable retention insights.

---

## 🎯 Problem
Customer churn directly impacts **MRR and LTV** in SaaS companies.  
This project helps identify **high-risk customers early**, understand churn drivers, and support data-driven retention strategies.

---

## 📂 Dataset
- **Source:** Kaggle – Telco Customer Churn  
- **Target:** `Churn` (Yes / No)

---

## 🧠 Solution
- Binary classification using **Random Forest**
- Feature engineering with SaaS-style metrics
- **Explainable AI (SHAP)** for customer-level churn reasons
- Interactive **Streamlit dashboard** with:
  - Risk segmentation
  - Visual analytics
  - Retention recommendations
  - Revenue impact estimation

---

## 🛠️ Tech Stack
- Python, Pandas, NumPy  
- Scikit-learn  
- SHAP  
- Streamlit, Plotly  

---

## ▶️ Run the Project
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app_master.py