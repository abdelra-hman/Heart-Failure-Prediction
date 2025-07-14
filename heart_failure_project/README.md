---
title: Heart Failure Predictor
emoji: 🫀
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: "4.20.0"
app_file: app.py
pinned: false
---

# 🫀 Heart Failure Prediction with Random Forest

This project predicts whether a patient is likely to experience heart failure based on clinical records using a **Random Forest Classifier**.  
It includes data preprocessing, SMOTE balancing, model training, evaluation, and a simple Gradio interface for predictions.

---

## 📁 Dataset

- **Source**: `heart_failure_clinical_records_dataset.csv`
- **Target Variable**: `DEATH_EVENT` (1 = death, 0 = survived)
- **Features**:
  - age
  - anaemia
  - creatinine_phosphokinase
  - diabetes
  - ejection_fraction
  - high_blood_pressure
  - platelets
  - serum_creatinine
  - serum_sodium
  - sex
  - smoking
  - time
  - + 2 new engineered features: `platelets_per_age`, `creatinine_per_ck`

---

## 🧠 Model Used

- RandomForestClassifier
- SMOTE (to handle data imbalance)
- StandardScaler (for feature scaling)

---

## 🧪 Evaluation

- 🎯 Accuracy: **95.08%**
- ✅ Stratified CV Mean Accuracy: **83.8%**

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
