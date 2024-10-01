from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the pre-trained model
model_filename1 = 'models/gb_boost_model.pkl'
with open(model_filename1, 'rb') as file:
    model_gb_boost = pickle.load(file)

model_filename2 = 'models/standard_scaler.pkl'
with open(model_filename2, 'rb') as f:
    scaler = pickle.load(f)

# Try loading the feature columns and handle any potential errors
try:
    with open('models/feature_columns.pkl', 'rb') as file:
        feature_columns = pickle.load(file)
except FileNotFoundError:
    feature_columns = None  # Handle missing file gracefully
    print("Error: 'feature_columns.pkl' not found. Make sure it is available in the correct directory.")
except Exception as e:
    feature_columns = None  # Handle other file loading errors
    print(f"Error loading 'feature_columns.pkl': {e}")

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and predict churn
@app.route('/predict', methods=['POST'])
def predict():
    
    form_data = request.form

    TotalCharges = pd.to_numeric(form_data['MonthlyCharges'], errors='coerce')
    
    # Convert form data to a dataframe
    input_data = pd.DataFrame([{
        'gender': form_data['gender'],
        'SeniorCitizen': int(form_data['SeniorCitizen']),
        'Partner': form_data['Partner'],
        'Dependents': form_data.get('Dependents', 'No'),
        'tenure': int(form_data['tenure']),
        'PhoneService': form_data['PhoneService'],
        'MultipleLines': form_data['MultipleLines'],
        'InternetService': form_data['InternetService'],
        'OnlineSecurity': form_data['OnlineSecurity'],
        'OnlineBackup': form_data['OnlineBackup'],
        'DeviceProtection': form_data['DeviceProtection'],
        'TechSupport': form_data['TechSupport'],
        'StreamingTV': form_data['StreamingTV'],
        'StreamingMovies': form_data['StreamingMovies'],
        'Contract': form_data['Contract'],
        'PaperlessBilling': form_data['PaperlessBilling'],
        'PaymentMethod': form_data['PaymentMethod'],
        'MonthlyCharges': float(form_data['MonthlyCharges']),
        'TotalCharges': TotalCharges
    }])

    # Apply one-hot encoding
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    input_data = pd.get_dummies(input_data, columns=cat_cols)
    
    # List of columns your model was trained with
    # required_columns = [
    #     'gender_Female', 'gender_Male', 'SeniorCitizen', 'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes',
    #     'tenure', 'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_Yes',
    #     'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    #     'OnlineSecurity_No', 'OnlineSecurity_Yes', 'OnlineBackup_No',
    #     'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_Yes',
    #     'TechSupport_No', 'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_Yes', 
    #     'StreamingMovies_No', 'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
    #     'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes', 'PaymentMethod_Electronic check',
    #     'PaymentMethod_Mailed check', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    #     'MonthlyCharges', 'TotalCharges'
    # ]

    required_columns = [
        'SeniorCitizen',
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'gender_Female',
        'gender_Male',
        'InternetService_DSL',
        'InternetService_Fiber optic',
        'InternetService_No',
        'Contract_Month-to-month',
        'Contract_One year',
        'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
        'Partner_No',
        'Partner_Yes',
        'Dependents_No',
        'Dependents_Yes',
        'PhoneService_No',
        'PhoneService_Yes',
        'PaperlessBilling_No',
        'PaperlessBilling_Yes',
        'MultipleLines_No',
        'MultipleLines_Yes',
        'OnlineSecurity_No',
        'OnlineSecurity_Yes',
        'OnlineBackup_No',
        'OnlineBackup_Yes',
        'DeviceProtection_No',
        'DeviceProtection_Yes',
        'TechSupport_No',
        'TechSupport_Yes',
        'StreamingTV_No',
        'StreamingTV_Yes',
        'StreamingMovies_No',
        'StreamingMovies_Yes',
        'tenure_range',
        'short_contract',
        'electronic_check'
    ]
    
    # Add missing columns to the input data and set them to 0
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    

    # Reorder the input columns to match the training order
    input_data = input_data[feature_columns]

    # Feature Engineering
    input_data['tenure_range'] = np.select(
        [(input_data['tenure'] >= 0) & (input_data['tenure'] <= 12),
         (input_data['tenure'] > 12) & (input_data['tenure'] <= 24),
         (input_data['tenure'] > 24) & (input_data['tenure'] <= 36),
         (input_data['tenure'] > 36) & (input_data['tenure'] <= 48),
         (input_data['tenure'] > 48) & (input_data['tenure'] <= 60),
         (input_data['tenure'] > 60)],
        [0, 1, 2, 3, 4, 5])

    input_data['short_contract'] = (input_data.get('Contract_Month-to-month', 0) == 1).astype(int)
    input_data['electronic_check'] = input_data.get('PaymentMethod_Electronic check', 0)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn using the model
    prediction = model_gb_boost.predict(input_data_scaled)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
