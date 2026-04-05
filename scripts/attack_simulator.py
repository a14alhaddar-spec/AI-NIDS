"""
Attack Simulator - Send Test Attack Traffic to Dashboards
Simulates various attack patterns and sends to inference services
Perfect for testing without network capture
"""
import requests
import time
import random
from datetime import datetime

# Inference services
INFERENCE_SERVICES = {
    'Random Forest': 'http://localhost:5002/infer',
    'CNN-LSTM': 'http://localhost:5003/infer',
    'CNN': 'http://localhost:5004/infer',
    'LSTM': 'http://localhost:5005/infer'
}

# Attack patterns based on CICIDS2017 signatures
ATTACK_PATTERNS = {
    'benign': {
        'Destination Port': 80,
        'Flow Duration': 5000,
        'Total Fwd Packets': 10,
        'Total Backward Packets': 8,
        'Total Length of Fwd Packets': 500,
        'Total Length of Bwd Packets': 400,
        'Flow Bytes/s': 180
    },
    'ddos': {
        'Destination Port': 80,
        'Flow Duration': 1000,
        'Total Fwd Packets': 5000,
        'Total Backward Packets': 0,
        'Total Length of Fwd Packets': 250000,
        'Total Length of Bwd Packets': 0,
        'Flow Bytes/s': 250000
    },
    'port_scan': {
        'Destination Port': 22,
        'Flow Duration': 100,
        'Total Fwd Packets': 1,
        'Total Backward Packets': 0,
        'Total Length of Fwd Packets': 60,
        'Total Length of Bwd Packets': 0,
        'Flow Bytes/s': 600
    },
    'botnet': {
        'Destination Port': 6667,
        'Flow Duration': 120000,
        'Total Fwd Packets': 150,
        'Total Backward Packets': 140,
        'Total Length of Fwd Packets': 7500,
        'Total Length of Bwd Packets': 70000,
        'Flow Bytes/s': 645
    },
    'web_attack': {
        'Destination Port': 80,
        'Flow Duration': 3000,
        'Total Fwd Packets': 45,
        'Total Backward Packets': 12,
        'Total Length of Fwd Packets': 5000,
        'Total Length of Bwd Packets': 1200,
        'Flow Bytes/s': 2066
    },
    'brute_force': {
        'Destination Port': 22,
        'Flow Duration': 500,
        'Total Fwd Packets': 8,
        'Total Backward Packets': 6,
        'Total Length of Fwd Packets': 320,
        'Total Length of Bwd Packets': 240,
        'Flow Bytes/s': 1120
    }
}

def send_attack(attack_type, features):
    """Send attack pattern to all inference services"""
    print(f"\n{'='*70}")
    print(f"🎯 Sending {attack_type.upper().replace('_', ' ')} attack pattern")
    print(f"{'='*70}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 Features:")
    for key, value in features.items():
        print(f"   {key}: {value}")
    
    results = {}
    threat_detected = False
    
    print(f"\n🔍 Inference Results:")
    for model_name, url in INFERENCE_SERVICES.items():
        try:
            response = requests.post(url, json=features, timeout=3)
            if response.status_code == 200:
                result = response.json()
                results[model_name] = result
                
                threat = "🚨 THREAT" if result.get('threat_detected') else "✅ BENIGN"
                threat_type = result.get('threat_type', 'Unknown')
                confidence = result.get('confidence', 0) * 100
                latency = result.get('inference_time_ms', 0)
                
                print(f"   {model_name:15} {threat:12} {threat_type:15} {confidence:5.1f}% ({latency:.1f}ms)")
                
                if result.get('threat_detected'):
                    threat_detected = True
            else:
                print(f"   {model_name:15} ❌ ERROR (HTTP {response.status_code})")
        except requests.exceptions.ConnectionError:
            print(f"   {model_name:15} ⚠️  Service not running")
        except Exception as e:
            print(f"   {model_name:15} ❌ ERROR ({str(e)[:30]})")
    
    if threat_detected:
        print(f"\n🚨 ATTACK DETECTED by one or more models!")
    else:
        print(f"\n✅ Traffic classified as benign by all models")
    
    return results

def check_services():
    """Check if inference services are running"""
    print("\n" + "="*70)
    print("CHECKING INFERENCE SERVICES")
    print("="*70)
    
    services_ok = 0
    for model_name, url in INFERENCE_SERVICES.items():
        try:
            health_url = url.replace('/infer', '/health')
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                print(f"✅ {model_name:15} - Running")
                services_ok += 1
            else:
                print(f"⚠️  {model_name:15} - Not responding")
        except:
            print(f"❌ {model_name:15} - Not accessible")
    
    print(f"\n{services_ok}/{len(INFERENCE_SERVICES)} services running")
    
    if services_ok == 0:
        print("\n❌ No services running! Start with: .\\START_ALL_DASHBOARDS.bat")
        return False
    
    return True

def run_attack_simulation():
    """Run interactive attack simulation"""
    print("\n" + "="*70)
    print("ATTACK SIMULATOR - AI-NIDS TEST TOOL")
    print("="*70)
    print("\nThis tool sends simulated attack patterns to your inference services")
    print("and displays detection results in real-time.\n")
    
    if not check_services():
        return
    
    print("\n" + "="*70)
    print("AVAILABLE ATTACK TYPES")
    print("="*70)
    attack_types = list(ATTACK_PATTERNS.keys())
    for i, attack in enumerate(attack_types, 1):
        print(f"{i}. {attack.upper().replace('_', ' ')}")
    print(f"{len(attack_types) + 1}. RUN ALL ATTACKS")
    print(f"{len(attack_types) + 2}. CONTINUOUS RANDOM ATTACKS")
    print("0. EXIT")
    
    while True:
        try:
            print("\n" + "="*70)
            choice = input("Select attack type (0 to exit): ").strip()
            
            if choice == '0':
                print("\n✅ Exiting...")
                break
            
            choice_num = int(choice)
            
            if choice_num == len(attack_types) + 1:
                # Run all attacks
                print("\n🚀 Running ALL attack types...")
                for attack_type in attack_types:
                    send_attack(attack_type, ATTACK_PATTERNS[attack_type])
                    time.sleep(2)
                print("\n✅ All attacks sent!")
                
            elif choice_num == len(attack_types) + 2:
                # Continuous random attacks
                print("\n🚀 Starting continuous random attack simulation...")
                print("Press Ctrl+C to stop\n")
                try:
                    count = 0
                    while True:
                        attack_type = random.choice(attack_types)
                        # Slightly randomize features
                        features = ATTACK_PATTERNS[attack_type].copy()
                        for key in features:
                            features[key] = int(features[key] * random.uniform(0.8, 1.2))
                        
                        send_attack(attack_type, features)
                        count += 1
                        print(f"\n💤 Waiting 3 seconds... ({count} attacks sent)")
                        time.sleep(3)
                except KeyboardInterrupt:
                    print(f"\n\n✅ Stopped after {count} attacks")
                
            elif 1 <= choice_num <= len(attack_types):
                # Send specific attack
                attack_type = attack_types[choice_num - 1]
                send_attack(attack_type, ATTACK_PATTERNS[attack_type])
            else:
                print("❌ Invalid choice")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n✅ Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == '__main__':
    run_attack_simulation()
