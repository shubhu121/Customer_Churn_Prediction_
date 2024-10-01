import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from pickle import dump
import os
import joblib

df = pd.read_csv("Curstomer Chrun.csv")

df.drop(index=359, inplace=True)

# 1. Binning Tenure

# Define bins and labels
bins = [0, 12, 24, 36, 48, 60, np.inf]
labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '60+']

# Create a binned column
df['Tenure_Group'] = pd.cut(df['tenure'], bins=bins, labels=labels)

# 2. Binning Monthly Charges

# Define bins and labels
bins = [0, 20, 40, 60, 80, 100, float('inf')]
labels = ['Very Low', 'Low', 'Medium', 'Medium-to-high', 'High', 'Very High']

# Create a binned column
df['MonthlyCharges_Binned'] = pd.cut(df['MonthlyCharges'], bins=bins, labels=labels)

# 3. Create 'AverageMonthlyCharge' feature
df['AverageMonthlyCharge'] = df['TotalCharges'] / df['tenure'].replace(0, np.nan)

# 4. Create 'Revenue_Contribution' feature
df['Revenue_Contribution'] = df['MonthlyCharges'] * df['tenure']

# Initialize OneHotEncoder with sparse_output=False
encoder = OneHotEncoder(sparse_output=False, drop='first')

# List of categorical features you want to encode
categorical_features = ['gender', 'Partner', 'Dependents', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 
                        'PaymentMethod', 'PhoneService']

# Fit and transform the categorical features
encoded_features = encoder.fit_transform(df[categorical_features])

# Get the encoded feature names for the OneHotEncoder
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

# Convert encoded features into a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Drop the original categorical columns from the original dataframe
df = df.drop(columns=categorical_features)

# Concatenate the original DataFrame with the new encoded features
df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

numerical_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']

# Fit encoders for each numerical column
for col in numerical_cols:
  scaler = StandardScaler()
  df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

binned_features = ['Tenure_Group', 'MonthlyCharges_Binned']

# Fit encoders for each binned column
label_encoder = LabelEncoder()

for col in binned_features:
  df[col] = label_encoder.fit_transform(df[col])

df['BothStreamingServices'] = ((df['StreamingTV_Yes'] == 1) & (df['StreamingMovies_Yes'] == 1)).astype(int)

encoder = OneHotEncoder(sparse_output=False)
df['Churn'] = encoder.fit_transform(df[['Churn']])

df.drop('customerID', axis=1, inplace=True)

X=df.drop('Churn', axis=1)
y=df['Churn']

'''
print(df.shape)
print(df.info())
print(X.shape)
print(y.shape)
'''
# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature selector
estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

selector = SelectBySingleFeaturePerformance(
	estimator=estimator,
	scoring="r2"
) 

# Fit the feature selector on the training data only
selector.fit(X_train, y_train)

# Transform both the training and validation sets using the fitted selector
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# print(X_train_selected.info())

# Define the directory where you want to save the preprocessor
save_dir = 'saved_models'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

joblib.dump(selector, os.path.join(save_dir, "preprocessor.joblib"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
joblib.dump(encoder, os.path.join(save_dir, "encoder.joblib"))
joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.joblib"))
print("Saved!!")