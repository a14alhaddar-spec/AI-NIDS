"""
Test All 4 Models - Send real CICIDS attack data to all inference services
"""

import pandas as pd
import requests
import json
import numpy as np
import time
# Request settings (models can be slow on first inference)
REQUEST_TIMEOUT = 60
WARMUP_TIMEOUT = 90
MAX_RETRIES = 1
RETRY_DELAY_SEC = 3


# Service URLs
SERVICES = {
    "Random Forest": "http://localhost:5002/infer",
    "CNN-LSTM": "http://localhost:5003/infer",
    "CNN": "http://localhost:5004/infer",
    "LSTM": "http://localhost:5005/infer"
}

# Key features to extract
KEY_FEATURES = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Flow Bytes/s'
]

def load_attack_samples():
    """Load PortScan samples from CICIDS2017"""
    print("Loading CICIDS2017 PortScan dataset...")
    df = pd.read_csv('datasets/CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    print(f"✅ Loaded {len(df)} samples with {len(df.columns)} features")
    print(f"Label distribution: {df['Label'].value_counts().to_dict()}")
    
    # Get Port Scan samples
    port_scan_samples = df[df['Label'] == 'PortScan'].sample(n=5, random_state=42)
    benign_samples = df[df['Label'] == 'BENIGN'].sample(n=2, random_state=42)
    
    return port_scan_samples, benign_samples

def extract_features(row):
    """Extract 7 key features from a sample"""
    features = {}
    for col in KEY_FEATURES:
        # Try with and without leading space
        val = row.get(col, row.get(' ' + col, row.get(col.strip(), 0.0)))
        
        # Handle NaN and infinity
        if pd.isna(val) or val == float('inf') or val == float('-inf'):
            val = 0.0
        
        features[col] = float(val)
    
    return features

def test_service(service_name, url, features):
    """Send features to a service and get prediction"""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = requests.post(url, json=features, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}"}
        except requests.exceptions.Timeout:
            last_error = f"Read timed out after {REQUEST_TIMEOUT}s (attempt {attempt})"
        except requests.exceptions.ConnectionError:
            last_error = "Service not running"
            break
        except Exception as e:
            last_error = str(e)

        if attempt <= MAX_RETRIES:
            time.sleep(RETRY_DELAY_SEC)

    return {"error": last_error or "Unknown error"}

def warmup_services(features):
    """Warm up models to avoid first-request latency"""
    print("Warming up inference services...")
    for service_name, url in SERVICES.items():
        try:
            requests.post(url, json=features, timeout=WARMUP_TIMEOUT)
            print(f"  ✅ {service_name}: warm-up ok")
        except Exception as e:
            print(f"  ⚠️  {service_name}: warm-up failed ({e})")

def print_separator(char="=", length=70):
    print(char * length)

def main():
    print_separator()
    print(" 🔍 AI-NIDS Multi-Model Detection Test")
    print_separator()
    print()
    
    # Load data
    port_scans, benign = load_attack_samples()
    print(f"Selected 5 PortScan + 2 Benign samples\n")
    
    # Warm up models with the first PortScan sample
    warmup_features = extract_features(port_scans.iloc[0])
    warmup_services(warmup_features)
    print()

    # Test Port Scans
    print_separator()
    print(" 🚨 TESTING PORT SCAN ATTACKS")
    print_separator()
    print()
    
    port_scan_detections = {name: 0 for name in SERVICES.keys()}
    
    for idx, (_, sample) in enumerate(port_scans.iterrows(), 1):
        features = extract_features(sample)
        print(f"Sample #{idx}:")
        print(f"  Destination Port: {features['Destination Port']}")
        print(f"  Flow Duration: {features['Flow Duration']}")
        print()
        
        for service_name, url in SERVICES.items():
            result = test_service(service_name, url, features)
            
            if "error" in result:
                print(f"  ❌ {service_name}: {result['error']}")
            else:
                prediction = result.get('prediction', 'Unknown')
                confidence = result.get('confidence', 0.0) * 100
                is_threat = result.get('threat_detected', False)
                
                if is_threat:
                    port_scan_detections[service_name] += 1
                    print(f"  ✅ {service_name}: {prediction} ({confidence:.1f}%) 🚨 THREAT")
                else:
                    print(f"  ⚠️  {service_name}: {prediction} ({confidence:.1f}%) - Missed")
        
        print()
    
    # Test Benign
    print_separator()
    print(" ✅ TESTING BENIGN TRAFFIC")
    print_separator()
    print()
    
    benign_correct = {name: 0 for name in SERVICES.keys()}
    
    for idx, (_, sample) in enumerate(benign.iterrows(), 1):
        features = extract_features(sample)
        print(f"Sample #{idx}:")
        print(f"  Destination Port: {features['Destination Port']}")
        print(f"  Flow Duration: {features['Flow Duration']}")
        print()
        
        for service_name, url in SERVICES.items():
            result = test_service(service_name, url, features)
            
            if "error" in result:
                print(f"  ❌ {service_name}: {result['error']}")
            else:
                prediction = result.get('prediction', 'Unknown')
                confidence = result.get('confidence', 0.0) * 100
                is_threat = result.get('threat_detected', False)
                
                if not is_threat:
                    benign_correct[service_name] += 1
                    print(f"  ✅ {service_name}: {prediction} ({confidence:.1f}%) - Correct")
                else:
                    print(f"  ⚠️  {service_name}: {prediction} ({confidence:.1f}%) - False Positive")
        
        print()
    
    # Summary
    print_separator("=", 70)
    print(" 📊 DETECTION SUMMARY")
    print_separator("=", 70)
    print()
    
    for service_name in SERVICES.keys():
        port_scan_rate = (port_scan_detections[service_name] / 5) * 100
        benign_rate = (benign_correct[service_name] / 2) * 100
        
        print(f"{service_name}:")
        print(f"  Port Scans Detected: {port_scan_detections[service_name]}/5 ({port_scan_rate:.0f}%)")
        print(f"  Benign Correct: {benign_correct[service_name]}/2 ({benign_rate:.0f}%)")
        print()
    
    print_separator("=", 70)

if __name__ == "__main__":
    main()
