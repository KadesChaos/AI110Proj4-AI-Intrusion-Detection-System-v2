# evaluate_system.py
"""Standalone evaluation harness: runs the trained detector stack against a fixed
set of predefined test batches and reports pass/fail for each, plus a summary.

This is separate from the runtime evaluation gate in train.py (which checks
whether a single retraining update should be accepted). This script instead
checks whether the system as a whole behaves as expected on known inputs, e.g.
does a normal batch stay quiet, does a heavily malicious batch get flagged.
"""
import os
import sys
import numpy as np
import pandas as pd

from data_handler import load_data, preprocess_data
from train import (train_models, train_anomaly_detector, detect_anomalies,
                    detect_malicious_activities, evaluate_model)
from models import initialize_tensorflow_model, initialize_malicious_detector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_PATH = os.path.join(BASE_DIR, 'UNSW_NB15_training-set.csv')
TESTING_PATH = os.path.join(BASE_DIR, 'UNSW_NB15_testing-set.csv')


def build_case_normal_only(test_data, n=100):
    rows = test_data[test_data['attack_cat'] == 'Normal'].head(n)
    return "Normal-only batch", rows, "no_anomaly"


def build_case_heavily_malicious(test_data, n=100):
    rows = test_data[test_data['attack_cat'].isin(['Generic', 'DoS', 'Exploits'])].head(n)
    return "Heavily malicious batch (Generic/DoS/Exploits)", rows, "anomaly_and_malicious"


def build_case_mixed(test_data, n=100):
    normal = test_data[test_data['attack_cat'] == 'Normal'].head(n // 2)
    attack = test_data[test_data['attack_cat'] == 'Reconnaissance'].head(n // 2)
    rows = pd.concat([normal, attack])
    return "Mixed batch (half Normal, half Reconnaissance)", rows, "anomaly_expected"


def run_case(name, rows, expectation, anomaly_detector, malicious_detector):
    features, _ = preprocess_data(rows, fit_scaler=False, is_train=False)
    anomalies = detect_anomalies(anomaly_detector, features)
    any_anomaly = bool(np.any(anomalies))
    any_malicious = bool(detect_malicious_activities(malicious_detector, features).any()) if any_anomaly else False

    if expectation == "no_anomaly":
        passed = not any_anomaly
        detail = f"anomaly_detected={any_anomaly} (expected False)"
    elif expectation == "anomaly_and_malicious":
        passed = any_anomaly and any_malicious
        detail = f"anomaly_detected={any_anomaly}, malicious_detected={any_malicious} (expected both True)"
    elif expectation == "anomaly_expected":
        passed = any_anomaly
        detail = f"anomaly_detected={any_anomaly} (expected True)"
    else:
        raise ValueError(f"Unknown expectation: {expectation}")

    return passed, detail


def main():
    print("Loading and preprocessing training data...")
    training_data = load_data(TRAINING_PATH)
    features, labels = preprocess_data(training_data, fit_scaler=True, is_train=True)

    print("Training models...")
    train_models(features, labels)

    print("Training anomaly + malicious detectors for evaluation...")
    anomaly_detector = train_anomaly_detector(features)
    malicious_detector = initialize_malicious_detector(features.shape[1])
    malicious_detector.fit(features, labels > 0, epochs=5, verbose=0)

    test_data = load_data(TESTING_PATH)

    cases = [
        build_case_normal_only(test_data),
        build_case_heavily_malicious(test_data),
        build_case_mixed(test_data),
    ]

    print("\n" + "=" * 60)
    print("RUNNING TEST CASES")
    print("=" * 60)

    results = []
    for name, rows, expectation in cases:
        passed, detail = run_case(name, rows, expectation, anomaly_detector, malicious_detector)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
        results.append((name, passed))

    n_passed = sum(1 for _, p in results if p)
    n_total = len(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'} - {name}")
    print(f"\n{n_passed}/{n_total} test cases passed.")

    sys.exit(0 if n_passed == n_total else 1)


if __name__ == "__main__":
    main()
