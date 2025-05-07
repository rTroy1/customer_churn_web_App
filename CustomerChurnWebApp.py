import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Load the trained model
loaded_model = pickle.load(open('C:/Users/user/Downloads/trained_model.sav', 'rb'))

# Create function for prediction
def customer_churn(customer_data):
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
            st.warning(f"No encoder found for column: {column}")

    # Making the prediction
    prediction = loaded_model.predict(customer_data_df)
    pred_prob = loaded_model.predict_proba(customer_data_df)

    # Output the prediction result
    if prediction[0] == 1:
        return 'No Churn'
    else:
        return "Churn"

# Main function to create Streamlit inputs
def main():
    # Title of the app
    st.title("Customer Churn Prediction")

    # Input fields for customer data
    gender = st.radio('Gender', ['Male', 'Female'])
    senior_citizen = st.radio('Senior Citizen', ['Yes', 'No'])
    partner = st.radio('Partner', ['Yes', 'No'])
    dependents = st.radio('Dependents', ['Yes', 'No'])
    tenure = st.number_input('Tenure (in months)', min_value=0, max_value=100, value=1)
    phone_service = st.radio('Phone Service', ['Yes', 'No'])
    multiple_lines = st.radio('Multiple Lines', ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.radio('Online Security', ['Yes', 'No'])
    online_backup = st.radio('Online Backup', ['Yes', 'No'])
    device_protection = st.radio('Device Protection', ['Yes', 'No'])
    tech_support = st.radio('Tech Support', ['Yes', 'No'])
    streaming_tv = st.radio('Streaming TV', ['Yes', 'No'])
    streaming_movies = st.radio('Streaming Movies', ['Yes', 'No'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.radio('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=29.85)
    total_charges = st.number_input('Total Charges', min_value=0.0, max_value=5000.0, value=29.85)

    # Create a dictionary with the inputs
    customer_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Predict the churn
    if st.button('Predict'):
        result = customer_churn(customer_data)
        st.write(f"The prediction is: {result}")
 
# Run the app
if __name__ == '__main__':
    main()

