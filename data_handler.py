# data_handler.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

scaler = StandardScaler()  # Initialize the scaler
feature_columns = None  # This will store all possible columns after initial one-hot encoding

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        print("\033[92m" + "Data loaded successfully.\n" + "\033[0m")
        return data
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None

def preprocess_data(data, fit_scaler=False, is_train=False):
    global feature_columns, scaler
    if data is not None:
        data = data.copy()
        data.ffill(inplace=True)
        data_encoded = pd.get_dummies(data)

        if is_train:
            feature_columns = data_encoded.columns  # Save the columns from the train data
        else:
            data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)

        if fit_scaler:
            scaler.fit(data_encoded.drop('label', axis=1))
            print("\nScaler fitted on initial data.\n\n")

        features = scaler.transform(data_encoded.drop('label', axis=1))
        labels = data_encoded['label'].values
        print("\033[92m" + "Data preprocessed successfully." + "\033[0m")
        return features, labels
    else:
        print("No data to preprocess.")
        return None, None

def data_split(features, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    print("\n\nData split into training and testing sets.\n")
    return X_train, X_test, y_train, y_test

def simulate_data_stream(data, batch_size=100):
    data_copy = data.copy()
    while True:
        for i in range(0, len(data_copy), batch_size):
            yield data_copy.iloc[i:i + batch_size]
        time.sleep(10)