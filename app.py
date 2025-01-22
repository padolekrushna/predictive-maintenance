import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Predictive Maintenance Application')
st.write("This app predicts machine failure and its type using a synthetic dataset.")

# Load and preprocess dataset
@st.cache_data
def load_data():
    data = pd.read_csv('data_processed.csv')
    return data

data = load_data()
st.write("### Dataset Preview")
st.dataframe(data.head())

# Train-test split
X = data.drop(['Machine failure', 'type_of_failure'], axis=1)
y_failure = data['Machine failure']
y_failure_type = data['type_of_failure']

X_train, X_test, y_train, y_test = train_test_split(X, y_failure, test_size=0.2, random_state=42)
X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X, y_failure_type, test_size=0.2, random_state=42)

# Train models
rf_failure = RandomForestClassifier()
rf_failure.fit(X_train, y_train)
rf_failure_type = RandomForestClassifier()
rf_failure_type.fit(X_train_type, y_train_type)

st.write("Models trained successfully.")

# Input Section
st.write("### Predict Machine Failure")
Type = st.selectbox("Select Product Type", ['Low (L)', 'Medium (M)', 'High (H)'])
Rotational_speed = st.slider("Rotational Speed [rpm]", float(X['Rotational speed [rpm]'].min()), float(X['Rotational speed [rpm]'].max()))
Torque = st.slider("Torque [Nm]", float(X['Torque [Nm]'].min()), float(X['Torque [Nm]'].max()))
Tool_wear = st.slider("Tool Wear [min]", float(X['Tool wear [min]'].min()), float(X['Tool wear [min]'].max()))
Air_temp = st.slider("Air Temperature [°C]", float(X['Air temperature [c]'].min()), float(X['Air temperature [c]'].max()))
Process_temp = st.slider("Process Temperature [°C]", float(X['Process temperature [c]'].min()), float(X['Process temperature [c]'].max()))

Type_encoded = {'L': 0, 'M': 1, 'H': 2}[Type]

input_data = [[Type_encoded, Rotational_speed, Torque, Tool_wear, Air_temp, Process_temp]]

if st.button("Predict Machine Failure"):
    failure_prediction = rf_failure.predict(input_data)[0]
    failure_type_prediction = rf_failure_type.predict(input_data)[0]

    st.write("### Prediction Results")
    st.write(f"Machine Failure: {'Yes' if failure_prediction else 'No'}")
    if failure_prediction:
        failure_types = {0: 'No Failure', 1: 'TWF', 2: 'HDF', 3: 'PWF', 4: 'OSF', 5: 'RNF'}
        st.write(f"Type of Failure: {failure_types[failure_type_prediction]}")
