import streamlit as st
import pickle
import numpy as np
import os

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="Salary_Prediction_Application",
    page_icon="💼",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------ #
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ model.pkl file not found. Please upload it to GitHub.")
        st.stop()

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model

model = load_model()

# ------------------ UI ------------------ #
st.title("💼 Salary Prediction Application")
st.write("Enter details to predict salary:")

# ⚠️ Update features according to your model
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

# ------------------ PREDICTION ------------------ #
if st.button("Predict Salary"):
    try:
        input_data = np.array([[feature1, feature2, feature3, feature4]])
        prediction = model.predict(input_data)

        st.success(f"💰 Predicted Salary: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
