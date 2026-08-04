# train.py
from models import initialize_tensorflow_model, initialize_malicious_detector, initialize_random_forest
from data_handler import data_split
import numpy as np
from tabulate import tabulate
from sklearn.ensemble import IsolationForest

def train_models(features, labels):
    X_train, X_test, y_train, y_test = data_split(features, labels)
    
    # Training Random Forest
    rf_model = initialize_random_forest()
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test) * 100  # Converting to percentage
    print("\033[92m" + f"Random Forest model trained successfully with Test Accuracy: {rf_accuracy:.2f}%\n" + "\033[0m")

    # Existing TensorFlow training
    tf_model = initialize_tensorflow_model(X_train.shape[1])
    tf_model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    test_loss, test_acc = tf_model.evaluate(X_test, y_test)
    print(f"TensorFlow Test Accuracy: {test_acc * 100:.2f}%, \033[93mTest Loss: {test_loss}" + "\033[0m")  # Converting to percentage

    # Training malicious activity detector
    malicious_model = initialize_malicious_detector(X_train.shape[1])
    malicious_model.fit(X_train, y_train > 0, epochs=5)
    print("\033[92m" + "\nMalicious activity detector trained successfully.\n" + "\033[0m")

def evaluate_model(model, val_features, val_labels):
    _, accuracy = model.evaluate(val_features, val_labels, verbose=0)
    return accuracy

def update_model(model, features, labels, val_features=None, val_labels=None, min_accuracy_delta=-0.02):
    """Retrain model on new data, but only keep the update if it doesn't regress
    accuracy on a held-out validation set by more than min_accuracy_delta.

    Without this gate, every streamed batch (even a bad or adversarial one) would
    permanently alter the live model with no check on whether it actually helped.
    """
    if val_features is None or val_labels is None:
        model.fit(features, labels, epochs=5, validation_split=0.2)
        print("\033[1m" + "Model updated with new data (no evaluation gate configured).\n" + "\033[0m")
        return True

    pre_update_weights = model.get_weights()
    baseline_accuracy = evaluate_model(model, val_features, val_labels)

    model.fit(features, labels, epochs=5, validation_split=0.2, verbose=0)
    new_accuracy = evaluate_model(model, val_features, val_labels)

    if new_accuracy < baseline_accuracy + min_accuracy_delta:
        model.set_weights(pre_update_weights)
        print("\033[91m" + f"Update REJECTED: validation accuracy would drop from "
              f"{baseline_accuracy:.4f} to {new_accuracy:.4f}. Rolled back to previous weights.\n" + "\033[0m")
        return False

    print("\033[1m" + f"Model updated with new data. Validation accuracy: "
          f"{baseline_accuracy:.4f} -> {new_accuracy:.4f}\n" + "\033[0m")
    return True

def train_anomaly_detector(features):
    isol_forest = IsolationForest(n_estimators=100, contamination=0.01)
    isol_forest.fit(features)
    return isol_forest

def detect_anomalies(detector, features):
    predictions = detector.predict(features)
    return predictions == -1

def detect_malicious_activities(detector, features):
    predictions = detector.predict(features)
    return predictions > 0.5

def describe_anomalies(features, anomalies_indices):
    headers = ["Index", "Feature", "Value", "Status"]
    table_data = []
    for index in anomalies_indices:
        anomaly = features[index]
        for i, value in enumerate(anomaly):
            feature_desc = feature_description(i)
            if is_malicious(value, i):
                # Red background with white text for high visibility
                formatted_index = f"\033[41m\033[97m{index}\033[0m"  # Red background, white text
                formatted_feature = f"\033[41m\033[97m{feature_desc}\033[0m"
                formatted_value = f"\033[41m\033[97m{value:.2f}\033[0m"
                status = f"\033[41m\033[97mCRITICAL\033[0m"  # End coloring after the status
            else:
                formatted_index = str(index)
                formatted_feature = feature_desc
                formatted_value = f"{value:.2f}"
                status = "OK"
            table_data.append([formatted_index, formatted_feature, formatted_value, status])

    # Generate and return a single table for all anomalies detected
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    return f"\nMalicious activity detected:\n{table}"

def feature_description(index):
    # Map index to readable text descriptions
    descriptions = {
        0: "CPU Load",
        1: "Network Traffic Volume",
        2: "Memory Usage",
        3: "Number of Login Attempts",
        # Continue for additional features
    }
    return descriptions.get(index, f"Feature {index}")

def is_malicious(value, index):
    # Define criteria for what constitutes a malicious value
    thresholds = {
        0: 80,  # CPU Load > 80%
        1: 10000,  # Network traffic > 10,000 packets per minute
        2: 90,  # Memory usage > 90%
        3: 10,  # More than 10 login attempts
        # Define thresholds for additional features
    }
    return value > thresholds.get(index, float('inf'))
