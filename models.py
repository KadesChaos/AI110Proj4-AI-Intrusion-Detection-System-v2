# models.py
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

def initialize_tensorflow_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_dim,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("\033[92m" + "\nTensorFlow model initialized and compiled.\n" + "\033[0m")
    return model

def initialize_malicious_detector(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_dim,)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("\033[92m" + "\nMalicious detector model initialized and compiled.\n" + "\033[0m")
    return model

def initialize_random_forest():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("\033[92m" + "\nRandom Forest model initialized.\n" + "\033[0m")
    return model
