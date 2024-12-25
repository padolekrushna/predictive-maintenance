import streamlit as st
import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("ML Model Deployment with Streamlit")
st.write("This app predicts outcomes based on user input.")

# Input fields
input_features = []
num_features = 5  # Replace with the actual number of features your model expects
for i in range(num_features):
    value = st.number_input(f"Feature {i + 1}", value=0.0)
    input_features.append(value)

# Predict
if st.button("Predict"):
    prediction = model.predict(np.array(input_features).reshape(1, -1))
    st.success(f"Prediction: {prediction[0]}")
