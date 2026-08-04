# ids_system.py
import numpy as np
import sys
import os
import signal
import logging
import re
from train import (train_models, update_model, train_anomaly_detector, detect_anomalies,
                   detect_malicious_activities, describe_anomalies)
from data_handler import load_data, preprocess_data, simulate_data_stream
from models import initialize_tensorflow_model, initialize_malicious_detector
from datetime import datetime

# Helper function to strip ANSI escape codes
def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Custom formatter for logging that strips ANSI codes
class CleanFormatter(logging.Formatter):
    def format(self, record):
        original = super(CleanFormatter, self).format(record)
        return strip_ansi_codes(original)

# Setup logging to file with no ANSI codes and console with ANSI codes
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_directory = os.path.expanduser('~/Downloads')
    log_filename = f"ids_system_{timestamp}.log"
    log_file_path = os.path.join(log_directory, log_filename)

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Define a formatter that strips ANSI codes for file logging
    file_formatter = CleanFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # Define a formatter for console that does not strip ANSI codes
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File handler with the custom formatter for logs
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(file_formatter)

    # Console handler with standard formatter (retains ANSI colors)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # Set up the root logger with both handlers
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    logging.info(f"\nLogging to {log_file_path}")

setup_logging()

def signal_handler(signum, frame):
    logging.info("Received shutdown signal. Cleaning up...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_ids():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_filepath = os.path.join(base_dir, 'UNSW_NB15_training-set.csv')
    testing_filepath = os.path.join(base_dir, 'UNSW_NB15_testing-set.csv')
    
    logging.info("Starting training with: {}".format(training_filepath))
    training_data = load_data(training_filepath)
    features, labels = preprocess_data(training_data, fit_scaler=True, is_train=True)
    train_models(features, labels)
    
    logging.info("Training completed. Starting continuous monitoring...")
    run_continuous_ids(testing_filepath)

def run_continuous_ids(testing_filepath):
    test_data = load_data(testing_filepath)
    features, labels = preprocess_data(test_data, fit_scaler=False, is_train=False)
    model = initialize_tensorflow_model(features.shape[1])
    anomaly_detector = train_anomaly_detector(features)
    malicious_detector = initialize_malicious_detector(features.shape[1])
    
    try:
        for new_data in simulate_data_stream(test_data):
            new_features, _ = preprocess_data(new_data, fit_scaler=False, is_train=False)
            anomalies = detect_anomalies(anomaly_detector, new_features)
            if np.any(anomalies):
                anomalies_indices = np.where(anomalies)[0]
                anomalies_details = describe_anomalies(new_features, anomalies_indices)
                if detect_malicious_activities(malicious_detector, new_features).any():
                    logging.warning("Malicious activity DETECTED! Details: {}".format(anomalies_details))
                update_model(model, new_features, _)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user, shutting down.")
    finally:
        logging.info("Cleaning up resources...")

if __name__ == "__main__":
    run_ids()
