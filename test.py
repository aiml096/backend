import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
try:
    model = pickle.load(open("boston_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")

# Streamlit UI
st.title("🏡 Boston Housing Price Prediction")
st.write("Enter the house features to predict the price.")

# Define input fields dynamically
num_features = 13  # Adjust based on your model
inputs = []
for i in range(num_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

# Predict Button
if st.button("Predict Price"):
    try:
        new_data = scaler.transform(np.array(inputs).reshape(1, -1))
        output = model.predict(new_data)[0]
        st.success(f"🏠 Predicted House Price: ${output:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
