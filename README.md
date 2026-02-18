# 🎓 Student Placement Predictor

A Machine Learning web application that predicts whether a student is likely to be placed based on their CGPA and IQ score.

This project demonstrates the implementation of Logistic Regression with feature scaling and deployment using Streamlit.

---

## 🚀 Live Features

- Predict placement status (Placed / Not Placed)
- Shows probability score
- Interactive sliders for CGPA and IQ
- Decision boundary visualization
- Clean and responsive Streamlit UI

---

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression
- Preprocessing: StandardScaler
- Input Features:
  - CGPA (0–10)
  - IQ (0–200)
- Output:
  - 1 → Placed
  - 0 → Not Placed

The model was trained on the `placement.csv` dataset and exported using pickle.

---

## 📊 Model Interpretation

After training, the model learned:

- CGPA has strong positive influence on placement
- IQ has comparatively lower influence
- Decision boundary is mostly vertical, indicating CGPA is the dominant feature

---

