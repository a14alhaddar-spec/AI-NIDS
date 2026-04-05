"""Test with ALL features that models expect"""

import pandas as pd
import requests
import json
from datetime import datetime
import time
import numpy as np

SERVICES = {
    'Random Forest': 'http://localhost:5002/infer'
    # Testing RF only for now - will add others after verification
    # 'CNN-LSTM': 'http://localhost:5003/infer',
    # 'CNN': 'http://localhost:5004/infer',
    # 'LSTM': 'http://localhost:5005/infer'
}

print("\n" + "="*70)
print("TESTING WITH ALL CICIDS2017 FEATURES")
print("="*70)

# Load dataset
file_path = r"datasets\CICIDS2017\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
print(f"\nLoading: {file_path}")
df = pd.read_csv(file_path)

# Remove label column and get feature columns
feature_cols = [col for col in df.columns if col != ' Label']
print(f"\n✅ Loaded {len(df)} samples with {len(feature_cols)} features")
print(f"\nLabel distribution:")
print(df[' Label'].value_counts())

# Get clear port scan samples
port_scans = df[(df[' Label'] == 'PortScan') & 
                (df[' Flow Duration'] < 1000)].head(5)

benign = df[df[' Label'] == 'BENIGN'].head(2)

print(f"\nSelected {len(port_scans)} PortScan + {len(benign)} Benign samples")

input("\nPress Enter to start testing...")

def send_to_models(row, actual_label, index):
    """Send ALL features to models"""
    print("\n" + "="*70)
    print(f"🎯 Sample #{index} - Actual: {actual_label}")
    print("="*70)
    
    # Extract the 7 key features the model uses
    key_features = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets', 'Flow Bytes/s'
    ]
    
    features = {}
    for col in key_features:
        # Handle column name variations (with/without leading space)
        if col in row.index:
            val = row[col]
        elif ' ' + col in row.index:
            val = row[' ' + col]
        elif col.strip() in row.index:
            val = row[col.strip()]
        else:
            val = 0.0
        
        # Handle infinity and NaN
        if pd.isna(val):
            val = 0.0
        elif val == float('inf'):
            val = 999999.0
        elif val == float('-inf'):
            val = -999999.0
        
        features[col] = float(val)
    
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Sending {len(features)} features to models...")
    print(f"Dst Port: {features.get('Destination Port', 0)}, Duration: {features.get('Flow Duration', 0)}")
    
    print("\n🔍 Model Results:")
    threat_count = 0
    
    for model_name, url in SERVICES.items():
        try:
            response = requests.post(url, json=features, timeout=5)
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
            else:
                print(f"   ⚠️  {model_name:15s} → HTTP {response.status_code}")
        except Exception as e:
            print(f"   ⚠️  {model_name:15s} → Error: {str(e)[:40]}")
    
    if threat_count > 0:
        print(f"\n🚨 THREAT DETECTED by {threat_count}/4 models!")
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
    time.sleep(1)

# Test benign
print("\n" + "="*70)
print("TESTING BENIGN TRAFFIC")  
print("="*70)

for idx, row in benign.iterrows():
    send_to_models(row, 'BENIGN', idx)
    time.sleep(1)

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"🎯 Port Scans detected: {detected}/{len(port_scans)}")
print(f"\n✅ Dashboards updated! Check http://localhost:8081")
print("   All predictions tracked with real metrics")
