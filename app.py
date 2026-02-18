import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------
# Load Model
# ------------------------------
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ------------------------------
# Custom Styling
# ------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Title Section
# ------------------------------
st.title("🎓 Student Placement Prediction System")
st.write("Predict whether a student will be placed based on CGPA and IQ.")

# ------------------------------
# Layout Columns
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Enter Student Details")

    cgpa = st.slider("CGPA", 0.0, 10.0, 5.0)
    iq = st.slider("IQ", 0, 200, 100)

    predict_button = st.button("Predict Placement")

# ------------------------------
# Prediction Section
# ------------------------------
with col2:
    if predict_button:

        input_data = np.array([[cgpa, iq]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)


        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:
            st.success("✅ Student is likely to be Placed")
        else:
            st.error("❌ Student is Not Likely to be Placed")

        st.write(f"**Probability of Placement:** {probability[0][1]*100:.2f}%")
        st.progress(int(probability[0][1]*100))


# ------------------------------
# Decision Boundary Visualization
# ------------------------------
st.subheader("📈 Decision Boundary Visualization")

# Generate grid (RAW values)
x_min, x_max = 0, 10
y_min, y_max = 0, 200

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# ✅ SCALE THE GRID (VERY IMPORTANT)
grid_scaled = scaler.transform(grid)

Z = model.predict(grid_scaled)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()

ax.contourf(xx, yy, Z, alpha=0.3)

# Plot the current input point
ax.scatter(cgpa, iq, color='red', s=100, label="Input Student")

ax.set_xlabel("CGPA")
ax.set_ylabel("IQ")
ax.set_title("Logistic Regression Decision Boundary")
ax.legend()

st.pyplot(fig)

st.write("Model Coefficients:", model.coef_)
st.write("Model Intercept:", model.intercept_)
