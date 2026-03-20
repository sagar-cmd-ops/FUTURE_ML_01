
import streamlit as st
import numpy as np
from model import run_model

st.title("📊 Sales Prediction App")

model, df, mae, r2 = run_model()

st.write("Model Performance:")
st.write(f"MAE: {mae}")
st.write(f"R2 Score: {r2}")

st.subheader("Enter Date")

year = st.number_input("Year", 2010, 2030)
month = st.number_input("Month", 1, 12)
day = st.number_input("Day", 1, 31)

if st.button("Predict"):
    input_data = np.array([[year, month, day]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: ₹{prediction[0]:.2f}")
