"""Quick test to see which RF service is responding"""

import requests

url = "http://localhost:5002/health"

try:
    response = requests.get(url, timeout=2)
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ RF Service Health Check:")
        print(f"   Status: {data.get('status')}")
        print(f"   Model: {data.get('model', 'not specified')}")
        print(f"   Response: {data}")
    else:
        print(f"❌ HTTP {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")
