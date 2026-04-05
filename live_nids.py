import numpy as np
import time
import joblib
import tensorflow as tf
from scapy.all import sniff
from collections import deque

# ==============================
# Load Models
# ==============================

print("\nLoading AI NIDS Models...\n")

rf_model = joblib.load("models/cicids_full/rf_model.joblib")
cnn_model = tf.keras.models.load_model("models/cicids_full/cnn_model.h5")
lstm_model = tf.keras.models.load_model("models/cicids_full/lstm_model.h5")
hybrid_model = tf.keras.models.load_model("models/cicids_full/cnn_lstm_model.h5")

print("Models Loaded Successfully\n")

# ==============================
# Attack Labels
# ==============================

attack_labels = [
    "BENIGN",
    "DDoS",
    "PortScan",
    "Bot",
    "Infiltration",
    "Web Attack Brute Force",
    "Web Attack XSS",
    "Web Attack SQL Injection",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS Slowloris",
    "DoS Slowhttptest"
]

# ==============================
# Traffic Buffer
# ==============================

packet_buffer = deque(maxlen=500)
flow_window = 200

# ==============================
# Feature Extraction
# ==============================

def extract_features(packets):

    lengths = []
    protocols = []
    times = []

    for p in packets:
        lengths.append(len(p))
        times.append(p.time)

        if hasattr(p, "proto"):
            protocols.append(p.proto)
        else:
            protocols.append(0)

    if len(lengths) < 5:
        return None

    duration = max(times) - min(times)

    tcp = protocols.count(6)
    udp = protocols.count(17)
    icmp = protocols.count(1)

    features = [

        len(packets),                    # packet count
        np.mean(lengths),
        np.std(lengths),
        np.max(lengths),
        np.min(lengths),

        duration,
        sum(lengths),
        np.var(lengths),

        tcp,
        udp,
        icmp,

        tcp/len(packets),
        udp/len(packets),
        icmp/len(packets),

        np.percentile(lengths,25),
        np.percentile(lengths,50),
        np.percentile(lengths,75),

        np.mean(protocols),
        np.std(protocols),

        sum(lengths)/(duration+0.0001)

    ]

    return np.array(features)

# ==============================
# Prediction Function
# ==============================

def detect_attack(features):

    features = features.reshape(1,-1)
    features_dl = np.expand_dims(features, axis=2)

    rf_pred = rf_model.predict(features)[0]

    cnn_probs = cnn_model.predict(features_dl)[0]
    lstm_probs = lstm_model.predict(features_dl)[0]
    hybrid_probs = hybrid_model.predict(features_dl)[0]

    cnn_pred = np.argmax(cnn_probs)
    lstm_pred = np.argmax(lstm_probs)
    hybrid_pred = np.argmax(hybrid_probs)

    results = {
        "Random Forest": attack_labels[rf_pred],
        "CNN": attack_labels[cnn_pred],
        "LSTM": attack_labels[lstm_pred],
        "Hybrid CNN-LSTM": attack_labels[hybrid_pred]
    }

    confidence = {
        "CNN": round(np.max(cnn_probs)*100,2),
        "LSTM": round(np.max(lstm_probs)*100,2),
        "Hybrid": round(np.max(hybrid_probs)*100,2)
    }

    return results, confidence

# ==============================
# Packet Handler
# ==============================

def packet_callback(packet):

    packet_buffer.append(packet)

    if len(packet_buffer) >= flow_window:

        packets = list(packet_buffer)
        features = extract_features(packets)

        if features is None:
            return

        results, confidence = detect_attack(features)

        print("\n================ AI NIDS ALERT ================\n")

        for model, prediction in results.items():
            print(f"{model:20s}: {prediction}")

        print("\nConfidence Scores")

        for model, score in confidence.items():
            print(f"{model:10s}: {score}%")

        print("\n===============================================\n")

        packet_buffer.clear()

# ==============================
# Start Monitoring
# ==============================

print("Starting Real-Time Network Monitoring...")
print("Generate attacks from Kali to test detection\n")

sniff(prn=packet_callback, store=False)