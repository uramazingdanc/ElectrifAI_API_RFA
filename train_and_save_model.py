import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset (replace with the path to your data)
file_path = 'sample meter data v1 (test data).csv'  # Adjust path
data = pd.read_csv(file_path)

# Convert 'ts' to datetime and extract features
data['ts'] = pd.to_datetime(data['ts'], format='%Y%m%d%H%M%S')
data['year'] = data['ts'].dt.year
data['month'] = data['ts'].dt.month
data['day'] = data['ts'].dt.day
data['hour'] = data['ts'].dt.hour
data['minute'] = data['ts'].dt.minute

# Label encode 'meter_id' and 'record_type'
label_encoder = LabelEncoder()
data['meter_id'] = label_encoder.fit_transform(data['meter_id'])
data['record_type'] = label_encoder.fit_transform(data['record_type'])

# Normalize the 'KWH' column
scaler = MinMaxScaler()
data['KWH'] = scaler.fit_transform(data[['KWH']])

# Assume you have a target column 'anomaly' for classification
if 'anomaly' not in data.columns:
    data['anomaly'] = [1 if x > 0.5 else 0 for x in data['KWH']]  # Dummy binary target based on KWH threshold

# Separate the features and target variable
X = data.drop(columns=['ts', 'anomaly'])  # Remove 'ts' and target columns for training
y = data['anomaly']  # Target column

# Save the column names used during training
column_order = X.columns.tolist()
with open('column_order.pkl', 'wb') as f:
    pickle.dump(column_order, f)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the model, LabelEncoder, and MinMaxScaler using pickle
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model, LabelEncoder, MinMaxScaler, and column order have been saved.")
