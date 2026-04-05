"""Quick health check for all inference services"""

import requests

SERVICES = {
    'Random Forest (5002)': 'http://localhost:5002/health',
    'CNN-LSTM (5003)': 'http://localhost:5003/health',
    'CNN (5004)': 'http://localhost:5004/health',
    'LSTM (5005)': 'http://localhost:5005/health'
}

print("\n" + "="*60)
print("INFERENCE SERVICES HEALTH CHECK")
print("="*60 + "\n")

running = 0
for name, url in SERVICES.items():
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            print(f"✅ {name} - Running")
            running += 1
        else:
            print(f"⚠️  {name} - Error {response.status_code}")
    except:
        print(f"❌ {name} - Not accessible")

print(f"\n{running}/4 services are running\n")

if running == 4:
    print("✅ All services ready! Run test_real_attacks.py now")
else:
    print("⚠️  Wait 10 more seconds and check again")
