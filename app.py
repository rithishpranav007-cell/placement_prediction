import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Placement Prediction")

st.write("Enter student details")

cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0)

iq = st.number_input("IQ Score", min_value=50, max_value=200)

if st.button("Predict"):

    prediction = model.predict([[cgpa, iq]])

    if prediction[0] == 1:
        st.success("Student is likely to be Placed")
    else:
        st.error("Student is NOT likely to be Placed")