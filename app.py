import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("student.csv")

X = df.drop("pass", axis=1)
y = df["pass"]

model = LogisticRegression()
model.fit(X, y)

st.title("Student Pass Prediction")

h = st.number_input("Hours Studied")
a = st.number_input("Attendance %")
asmt = st.number_input("Assignments Done")

if st.button("Predict"):
    pred = model.predict([[h, a, asmt]])
    if pred[0] == 1:
        st.success("Student Will PASS")
    else:
        st.error("Student Will FAIL")
