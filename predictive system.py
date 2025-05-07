# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import pandas as pd  # Add this import to handle DataFrame
import pickle 

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/user/Downloads/trained_model.sav', 'rb'))

# Customer data dictionary
customer_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

# Convert customer data into DataFrame
customer_data_df = pd.DataFrame([customer_data])

# Load the saved encoders
with open("C:/Users/user/Downloads/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# List of categorical columns that need encoding
categorical_columns = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
    'PaymentMethod'
]

# Encode categorical features using the saved encoders
for column in categorical_columns:
    if column in encoders:
        encoder = encoders[column]
        customer_data_df[column] = encoder.transform(customer_data_df[column])
    else:
        print(f"Warning: No encoder found for column: {column}")

# Making the prediction
prediction = loaded_model.predict(customer_data_df)
pred_prob = loaded_model.predict_proba(customer_data_df)

# Output the prediction result
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Probability of Churn: {pred_prob}")
