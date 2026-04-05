"""Test with FULL feature set from CICIDS2017"""

import pandas as pd
import requests
import json
from datetime import datetime
import time

SERVICES = {
    'Random Forest': 'http://localhost:5002/infer',
    'CNN-LSTM': 'http://localhost:5003/infer',
    'CNN': 'http://localhost:5004/infer',
    'LSTM': 'http://localhost:5005/infer'
}

# All features the models expect (exact column names from CSV)
FEATURE_COLUMNS = [
    ' Destination Port',
    ' Flow Duration',
    ' Total Fwd Packets',
    ' Total Backward Packets',
    'Total Length of Fwd Packets',  # No leading space!
    ' Total Length of Bwd Packets',
    'Flow Bytes/s'  # No leading space!
]

print("\n" + "="*70)
print("TESTING WITH FULL CICIDS2017 FEATURES")
print("="*70)

# Load dataset
file_path = r"datasets\CICIDS2017\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
print(f"\nLoading: {file_path}")
df = pd.read_csv(file_path)

print(f"✅ Loaded {len(df)} samples")
print(f"\nLabel distribution:")
print(df[' Label'].value_counts())

# Get clear port scan samples (low duration, single packet)
port_scans = df[(df[' Label'] == 'PortScan') & 
                (df[' Flow Duration'] < 1000) &
                (df[' Total Fwd Packets'] <= 2)].head(10)

benign = df[df[' Label'] == 'BENIGN'].head(3)

print(f"\nSelected {len(port_scans)} PortScan + {len(benign)} Benign samples")

input("\nPress Enter to start testing...")

def send_to_models(row, actual_label, index):
    """Send sample to all models"""
    print("\n" + "="*70)
    print(f"🎯 Sample #{index} - Actual: {actual_label}")
    print("="*70)
    
    # Extract features
    features = {}
    for col in FEATURE_COLUMNS:
        val = row[col]
        # Handle infinity and NaN
        if pd.isna(val) or val == float('inf') or val == float('-inf'):
            val = 0
        features[col.strip()] = float(val)
    
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Dst Port: {features['Destination Port']}, Duration: {features['Flow Duration']}")
    print(f"Fwd Packets: {features['Total Fwd Packets']}, Bwd Packets: {features['Total Backward Packets']}")
    
    print("\n🔍 Model Results:")
    threat_count = 0
    
    for model_name, url in SERVICES.items():
        try:
            response = requests.post(url, json=features, timeout=3)
            if response.status_code == 200:
                result = response.json()
                pred = result.get('prediction', 'Unknown')
                conf = result.get('confidence', 0) * 100
                threat = result.get('threat_detected', False)
                
                if threat:
                    threat_count += 1
                    print(f"   🚨 {model_name:15s} → {pred:15s} ({conf:5.1f}%)")
                else:
                    print(f"   ✅ {model_name:15s} → {pred:15s} ({conf:5.1f}%)")
        except Exception as e:
            print(f"   ⚠️  {model_name:15s} → Error: {str(e)[:30]}")
    
    if threat_count > 0:
        print(f"\n🚨 THREAT DETECTED by {threat_count}/4 models")
    else:
        print("\n✅ All models: Benign")
    
    return threat_count

# Test port scans
print("\n" + "="*70)
print("TESTING PORT SCANS")
print("="*70)

detected = 0
for idx, row in port_scans.iterrows():
    if send_to_models(row, 'PortScan', idx) > 0:
        detected += 1
    time.sleep(0.5)

# Test benign
print("\n" + "="*70)
print("TESTING BENIGN TRAFFIC")  
print("="*70)

for idx, row in benign.iterrows():
    send_to_models(row, 'BENIGN', idx)
    time.sleep(0.5)

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"Port Scans detected: {detected}/{len(port_scans)}")
print("\n✅ Test complete!")
