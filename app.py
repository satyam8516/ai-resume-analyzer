
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("resume_model.pkl")

st.title("AI Resume Analyzer (ML + GenAI Demo)")

python = st.selectbox("Python Skill", [0,1])
sql = st.selectbox("SQL Skill", [0,1])
experience = st.slider("Years of Experience",0,10)
btech = st.selectbox("B.Tech Degree", [0,1])

if st.button("Analyze Resume"):
    data = pd.DataFrame([[python,sql,experience,btech]],
                        columns=["skills_python","skills_sql","experience_years","education_btech"])
    
    prediction = model.predict(data)[0]
    
    if prediction == 1:
        st.success("Good Resume: High chance of selection")
    else:
        st.error("Resume needs improvement")
    
    st.write("AI Suggestion: Add more technical projects and highlight relevant skills.")
