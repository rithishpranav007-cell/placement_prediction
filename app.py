import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Placement Prediction")

st.title("🎓 Placement Prediction")
st.write("Enter student details below")

# User Inputs
cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    step=0.1
)

internships = st.number_input(
    "Number of Internships",
    min_value=0,
    max_value=10,
    step=1
)

projects = st.number_input(
    "Number of Projects",
    min_value=0,
    max_value=20,
    step=1
)

aptitude_score = st.number_input(
    "Aptitude Score",
    min_value=0,
    max_value=100,
    step=1
)

# Prediction Button
if st.button("Predict Placement"):

    features = np.array([[cgpa, internships, projects, aptitude_score]])

    prediction = model.predict(features)

    probability = model.predict_proba(features)

    if prediction[0] == 1:
        st.success("🎉 Student is likely to be PLACED")
    else:
        st.error("❌ Student is NOT likely to be placed")

    st.write(f"Placement Probability: **{probability[0][1]*100:.2f}%**")
