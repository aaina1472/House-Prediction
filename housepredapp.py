# 🏠 Streamlit app for House Price Prediction

import streamlit as st
import numpy as np
from joblib import load
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ======== Load Model and Pipeline ========
model = load('Dragon_load.joblib')  # Your trained Decision Tree model
prediction = model.predict(input_array)


# IMPORTANT: Feature order must match training
feature_names = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS',
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Recreate the same preprocessing pipeline
preprocess_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# TEMP: Dummy fit the pipeline just to allow transform
# Better: Save and load your trained pipeline using joblib
dummy_data = np.random.rand(100, len(feature_names))
preprocess_pipeline.fit(dummy_data)

# ======== Streamlit UI ========
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 Boston House Price Prediction App")
st.write("Enter the house details below:")

# Input fields
input_array = np.array(inputs).reshape(1, -1)
inputs = []
for feature in feature_names:
    val = st.number_input(f"{feature}", step=0.1)
    inputs.append(val)

# Predict button
if st.button("Predict Price"):
    input_array = np.array(inputs).reshape(1, -1)
    input_prepared = preprocess_pipeline.transform(input_array)
    prediction = model.predict(input_prepared)
    st.success(f"💰 Predicted House Price: ${prediction[0]*1000:,.2f}")
