from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
# from model import ModelPreprocessor
import pickle
import os
import joblib
import logging

# Initialize Flask app
app = Flask(__name__)

save_dir = 'saved_models'
preprocessor = joblib.load(os.path.join(save_dir, "preprocessor.joblib"))
scaler = joblib.load(os.path.join(save_dir, "scaler.joblib"))
encoder = joblib.load(os.path.join(save_dir, "encoder.joblib"))
with open(os.path.join(save_dir, "random_forest_model.pkl"), 'rb') as file:
    model = pickle.load(file)

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)

# Render the home page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        data = request.form.to_dict()  # Convert form data to dictionary
        print("Incoming data:", data)

        # Log the incoming request data
        app.logger.debug("Incoming request data: %s", data)

        expected_columns = [
            'tenure', 
            'MonthlyCharges', 
            'TotalCharges', 
            'gender', 
            'Dependents', 
            'MultipleLines', 'Partner', 
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 
            'StreamingMovies', 'Contract', 'PaperlessBilling', 
            'PaymentMethod', 'PhoneService', 'SeniorCitizen'
        ]

        missing_columns = [col for col in expected_columns if col not in data]

        if missing_columns:
            return jsonify({'error': f'Missing input data for: {", ".join(missing_columns)}'}), 400

        # Convert the input data into a DataFrame
        input_data = pd.DataFrame(data, index=[0])

        # Reorder columns to match what was used in training
        expected_columns = preprocessor.get_feature_names_out()
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)

        print("Expected columns:", expected_columns)
        print("Current columns:", input_data.columns)

        app.logger.debug("DataFrame created: %s", input_data)

        missing_columns = [col for col in expected_columns if col not in input_data.columns]
        if missing_columns:
            print("Missing columns added:", missing_columns)

        input_data['tenure'] = input_data['tenure'].astype(float)
        input_data['MonthlyCharges'] = input_data['MonthlyCharges'].astype(float)
        input_data['TotalCharges'] = input_data['TotalCharges'].astype(float)

        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in input_data.columns:
                return jsonify({'error': f'Missing input data for: {col}'}), 400

        # Preprocess the input data
        input_data['Tenure_Group'] = pd.cut(input_data['tenure'], bins=[0, 12, 24, 36, 48, 60, np.inf], 
                                             labels=['0-12', '13-24', '25-36', '37-48', '49-60', '60+'])
        input_data['MonthlyCharges_Binned'] = pd.cut(input_data['MonthlyCharges'], 
                                                      bins=[0, 20, 40, 60, 80, 100, float('inf')], 
                                                      labels=['Very Low', 'Low', 'Medium', 'Medium-to-high', 
                                                              'High', 'Very High'])
        input_data['AverageMonthlyCharge'] = input_data['TotalCharges'] / input_data['tenure'].replace(0, np.nan)
        input_data['Revenue_Contribution'] = input_data['MonthlyCharges'] * input_data['tenure']
        input_data['BothStreamingServices'] = ((input_data['StreamingTV'] == '1') & 
                                                (input_data['StreamingMovies'] == '1')).astype(int)

        # Make sure to drop original categorical columns only if they exist
        categorical_features = ['gender', 'Partner', 'Dependents', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 
                        'PaymentMethod', 'PhoneService']
        
        # Encode categorical features using the encoder
        try:
            encoded_features = encoder.transform(input_data[categorical_features])
        except KeyError as e:
            return jsonify({'error': f'Missing categorical feature: {str(e)}'}), 400

        # Create DataFrame from the encoded features
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
        
        # Drop original categorical columns from input_data
        input_data = input_data.drop(columns=categorical_features, errors='ignore')
        
        # Concatenate the input data with encoded features
        input_data = pd.concat([input_data.reset_index(drop=True), encoded_df], axis=1)

        # Scale numerical features
        numerical_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # Select relevant features based on the preprocessor
        try:
            input_data_selected = preprocessor.transform(input_data)
        except KeyError as e:
            return jsonify({'error': f'Missing features after preprocessing: {str(e)}'}), 400

        # Perform prediction 
        prediction = model.predict(input_data_selected)  
        print(prediction)
        
        # Send back the prediction result
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

