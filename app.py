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
        st.error("❌ model.pkl not found. Upload it to your GitHub repo.")
        st.stop()

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model

model = load_model()

# ------------------ UI ------------------ #
st.title("💼 Salary Prediction Application")
st.write("Enter candidate details to predict salary")

# 🔥 Common Salary Prediction Features
experience = st.number_input("Years of Experience", min_value=0.0, step=0.5)
test_score = st.number_input("Test Score (out of 10)", min_value=0.0, max_value=10.0)
interview_score = st.number_input("Interview Score (out of 10)", min_value=0.0, max_value=10.0)

# ------------------ PREDICTION ------------------ #
if st.button("Predict Salary"):
    try:
        input_data = np.array([[experience, test_score, interview_score]])

        prediction = model.predict(input_data)

        st.success(f"💰 Predicted Salary: ₹ {round(prediction[0], 2)}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
