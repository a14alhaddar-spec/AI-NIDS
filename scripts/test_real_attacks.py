"""Test with selectable real attack samples from CICIDS2017 dataset."""

import os
import time
from datetime import datetime

import pandas as pd
import requests

# Inference services
SERVICES = {
    'Random Forest': 'http://localhost:5002/infer',
    'CNN-LSTM': 'http://localhost:5003/infer',
    'CNN': 'http://localhost:5004/infer',
    'LSTM': 'http://localhost:5005/infer'
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "CICIDS2017")
DATASET_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

FEATURE_MAP = {
    "Destination Port": " Destination Port",
    "Flow Duration": " Flow Duration",
    "Total Fwd Packets": " Total Fwd Packets",
    "Total Backward Packets": " Total Backward Packets",
    "Total Length of Fwd Packets": " Total Length of Fwd Packets",
    "Total Length of Bwd Packets": " Total Length of Bwd Packets",
    "Flow Bytes/s": " Flow Bytes/s",
}

BENIGN_LABELS = {"BENIGN", "NORMAL"}


def normalize_label(value):
    """Normalize labels for consistent filtering."""
    return str(value).strip()


def is_benign_label(label):
    """Check whether a label is benign/normal."""
    return normalize_label(label).upper() in BENIGN_LABELS


def to_number(value):
    """Convert potentially non-numeric values safely to a float/int-compatible number."""
    try:
        number = float(value)
        if pd.isna(number):
            return 0.0
        return number
    except Exception:
        return 0.0

def load_attack_catalog():
    """Scan CICIDS files and return available non-benign attack labels."""
    print("\n" + "="*70)
    print("SCANNING AVAILABLE CICIDS2017 ATTACK LABELS")
    print("="*70)

    catalog = {}
    for filename in DATASET_FILES:
        file_path = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(file_path):
            continue

        try:
            labels_df = pd.read_csv(file_path, usecols=[" Label"])
            labels = labels_df[" Label"].dropna().astype(str).map(normalize_label).unique()

            for label in labels:
                if is_benign_label(label):
                    continue
                catalog.setdefault(label, set()).add(file_path)
        except Exception as exc:
            print(f"⚠️  Failed to scan labels in {filename}: {exc}")

    if not catalog:
        print("❌ No attack labels were discovered in available CICIDS files")
        return {}

    print(f"✅ Found {len(catalog)} attack types across CICIDS files")
    return {label: sorted(list(paths)) for label, paths in catalog.items()}


def choose_attack_labels(catalog):
    """Interactive selector for attack labels."""
    labels = sorted(catalog.keys())

    print("\n" + "="*70)
    print("SELECT ATTACK TYPES TO TEST")
    print("="*70)
    for idx, label in enumerate(labels, 1):
        file_count = len(catalog[label])
        print(f"{idx:2}. {label} ({file_count} file{'s' if file_count != 1 else ''})")
    print(f"{len(labels) + 1:2}. ALL ATTACK TYPES")
    print(" 0. EXIT")

    while True:
        choice = input("\nSelect option(s) (e.g. 1 or 1,3,5): ").strip()

        if choice == "0":
            return []

        entries = [item.strip() for item in choice.split(",") if item.strip()]
        if not entries:
            print("❌ Please enter at least one selection")
            continue

        try:
            numbers = [int(item) for item in entries]
        except ValueError:
            print("❌ Use numbers only, separated by commas")
            continue

        all_option = len(labels) + 1
        if all_option in numbers:
            return labels

        valid_numbers = [number for number in numbers if 1 <= number <= len(labels)]
        if not valid_numbers:
            print("❌ Invalid selection")
            continue

        unique_sorted = sorted(set(valid_numbers))
        return [labels[number - 1] for number in unique_sorted]


def ask_sample_count(prompt, default):
    """Prompt for positive integer sample count."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("❌ Enter a positive integer")


def load_samples_for_label(label, file_paths, sample_count):
    """Load up to sample_count rows for a specific attack label from one or more files."""
    samples = []
    for file_path in file_paths:
        if len(samples) >= sample_count:
            break
        try:
            df = pd.read_csv(file_path)
            filtered = df[df[" Label"].astype(str).map(normalize_label) == label]
            for idx, row in filtered.iterrows():
                samples.append((idx, row, file_path))
                if len(samples) >= sample_count:
                    break
        except Exception as exc:
            print(f"⚠️  Failed to load samples from {os.path.basename(file_path)}: {exc}")
    return samples


def load_benign_samples(sample_count):
    """Load benign samples from the first available CICIDS files."""
    benign_samples = []
    for filename in DATASET_FILES:
        if len(benign_samples) >= sample_count:
            break
        file_path = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(file_path):
            continue
        try:
            df = pd.read_csv(file_path)
            filtered = df[df[" Label"].astype(str).map(normalize_label).map(is_benign_label)]
            for idx, row in filtered.iterrows():
                benign_samples.append((idx, row, file_path))
                if len(benign_samples) >= sample_count:
                    break
        except Exception as exc:
            print(f"⚠️  Failed to load benign samples from {filename}: {exc}")
    return benign_samples

def test_sample(features, label, index):
    """Send real sample to all models"""
    print("\n" + "="*70)
    print(f"🎯 Testing REAL SAMPLE #{index}")
    print("="*70)
    print(f"Actual Label: {label}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Prepare features (use only the 7 we need)
    feature_dict = {
        target_key: to_number(features.get(source_key, 0))
        for target_key, source_key in FEATURE_MAP.items()
    }
    
    print("\n📊 Features:")
    for key, val in feature_dict.items():
        print(f"   {key}: {val}")
    
    print("\n🔍 Model Predictions:")
    threat_detected = False
    
    for model_name, url in SERVICES.items():
        try:
            response = requests.post(url, json=feature_dict, timeout=3)
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', 'Unknown')
                confidence = result.get('confidence', 0) * 100
                
                is_threat = result.get('threat_detected', False)
                if is_threat:
                    threat_detected = True
                    icon = "🚨"
                else:
                    icon = "✅"
                
                print(f"   {icon} {model_name:15s} {prediction:15s} {confidence:5.1f}%")
            else:
                print(f"   ❌ {model_name:15s} HTTP {response.status_code}: {response.text[:80]}")
        except:
            print(f"   ⚠️  {model_name:15s} Service not running")
    
    if threat_detected:
        print("\n🚨 THREAT DETECTED by one or more models!")
    else:
        print("\n✅ Classified as benign by all models")
    
    return threat_detected

def main():
    print("\n" + "="*70)
    print("REAL ATTACK DATA TESTING")
    print("="*70)
    
    catalog = load_attack_catalog()
    if not catalog:
        return

    selected_labels = choose_attack_labels(catalog)
    if not selected_labels:
        print("\n✅ No attack type selected. Exiting...")
        return

    attack_samples_per_type = ask_sample_count("Attack samples per selected type", 5)
    benign_samples_count = ask_sample_count("Benign samples for baseline", 2)

    print("\n" + "=" * 70)
    print("TEST PLAN")
    print("=" * 70)
    print(f"Selected attack types: {', '.join(selected_labels)}")
    print(f"Samples per attack type: {attack_samples_per_type}")
    print(f"Benign samples: {benign_samples_count}")

    input("\nPress Enter to start testing...")

    total_attack_samples = 0
    total_attack_detected = 0

    for label in selected_labels:
        samples = load_samples_for_label(label, catalog[label], attack_samples_per_type)
        print("\n\n" + "=" * 70)
        print(f"TESTING ATTACK TYPE: {label}")
        print("=" * 70)

        if not samples:
            print(f"⚠️  No samples found for {label}")
            continue

        for idx, row, source_file in samples:
            print(f"Source file: {os.path.basename(source_file)}")
            detected = test_sample(row, label, idx)
            total_attack_samples += 1
            if detected:
                total_attack_detected += 1
            time.sleep(1)

    benign = load_benign_samples(benign_samples_count)
    benign_detected = 0

    print("\n\n" + "=" * 70)
    print("TESTING BENIGN TRAFFIC")
    print("=" * 70)

    for idx, row, source_file in benign:
        print(f"Source file: {os.path.basename(source_file)}")
        detected = test_sample(row, "BENIGN", idx)
        if detected:
            benign_detected += 1
        time.sleep(1)
    
    print("\n\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    if total_attack_samples > 0:
        print(f"\nThreats detected on attack traffic: {total_attack_detected}/{total_attack_samples}")
    else:
        print("\nNo attack samples were tested")
    print(f"False positives on benign traffic: {benign_detected}/{len(benign)}")
    print("\n✅ Check your dashboards to see the threat analytics!")
    print("   - Random Forest:  http://localhost:8081")
    print("   - CNN-LSTM:       http://localhost:8082")
    print("   - CNN:            http://localhost:8083")
    print("   - LSTM:           http://localhost:8084")

if __name__ == "__main__":
    main()
