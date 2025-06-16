import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('student-ml/student_model.pkl', 'rb'))

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict if they will pass:")

# Collect input data
age = st.slider("Age", 15, 22, 17)
studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
absences = st.slider("Absences", 0, 93, 5)
goout = st.slider("Going Out (1-5)", 1, 5, 3)
Dalc = st.slider("Weekday Alcohol Consumption (1-5)", 1, 5, 1)
Walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 2)

# extracting features
features = np.array([[age, studytime, failures, absences, goout, Dalc, Walc]])

if st.button("Predict"):
    result = model.predict(features)
    if result[0] == 1:
        st.success("✅ The student is likely to PASS!")
    else:
        st.error("❌ The student is likely to FAIL.")
