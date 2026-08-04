# data_handler.py
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

scaler = StandardScaler()  # Initialize the scaler
feature_columns = None  # This will store all possible columns after initial one-hot encoding
scaler_fitted = False  # Guards against silently refitting the scaler on new data

def load_data(filepath):
    data = pd.read_csv(filepath)
    logging.info("Data loaded successfully.")
    return data

def preprocess_data(data, fit_scaler=False, is_train=False, label_column='label'):
    global feature_columns, scaler, scaler_fitted
    if data is None:
        logging.warning("No data to preprocess.")
        return None, None

    data = data.copy()
    data.ffill(inplace=True)
    data.bfill(inplace=True)  # ffill alone leaves leading NaNs unfilled
    data_encoded = pd.get_dummies(data)

    if is_train:
        feature_columns = data_encoded.columns  # Save the columns from the train data
    else:
        if feature_columns is None:
            raise RuntimeError(
                "feature_columns is not set: call preprocess_data(..., is_train=True) "
                "on training data before preprocessing inference/streamed data."
            )
        data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)

    if fit_scaler:
        if scaler_fitted:
            logging.warning("Scaler was already fitted; refitting will invalidate previously trained models.")
        scaler.fit(data_encoded.drop(label_column, axis=1))
        scaler_fitted = True
        logging.info("Scaler fitted on initial data.")

    if not scaler_fitted:
        raise RuntimeError("Scaler is not fitted yet: call preprocess_data(..., fit_scaler=True) first.")

    features = scaler.transform(data_encoded.drop(label_column, axis=1))
    labels = data_encoded[label_column].values
    logging.info("Data preprocessed successfully.")
    return features, labels

def data_split(features, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    logging.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

def simulate_data_stream(data, batch_size=100, interval_seconds=10, stop_event=None):
    data_copy = data.copy()
    while stop_event is None or not stop_event.is_set():
        for i in range(0, len(data_copy), batch_size):
            if stop_event is not None and stop_event.is_set():
                return
            yield data_copy.iloc[i:i + batch_size]
        time.sleep(interval_seconds)
