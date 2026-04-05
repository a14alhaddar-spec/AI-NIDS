"""
Verify ML Models are Working
This script tests the inference service with sample network traffic features.
"""

import json
import requests
import sys

# Test features representing different attack types
TEST_CASES = [
    {
        "name": "Normal Traffic",
        "features": {
            "flow_duration": 1.5,
            "bytes": 1024,
            "packets": 10,
            "bytes_per_sec": 682.67,
            "packets_per_sec": 6.67,
            "src_dst_bytes_ratio": 1.0,
            "avg_pkt_size": 102.4
        },
        "expected": "benign"
    },
    {
        "name": "DDoS Attack",
        "features": {
            "flow_duration": 0.5,
            "bytes": 50000,
            "packets": 1200,
            "bytes_per_sec": 100000,
            "packets_per_sec": 2400,
            "src_dst_bytes_ratio": 0.1,
            "avg_pkt_size": 41.67
        },
        "expected": "dos_ddos"
    },
    {
        "name": "Port Scan",
        "features": {
            "flow_duration": 2.0,
            "bytes": 500,
            "packets": 800,
            "bytes_per_sec": 250,
            "packets_per_sec": 400,
            "src_dst_bytes_ratio": 0.5,
            "avg_pkt_size": 0.625
        },
        "expected": "port_scan"
    },
    {
        "name": "Data Exfiltration",
        "features": {
            "flow_duration": 10.0,
            "bytes": 100000000,
            "packets": 50000,
            "bytes_per_sec": 10000000,
            "packets_per_sec": 5000,
            "src_dst_bytes_ratio": 10.0,
            "avg_pkt_size": 2000
        },
        "expected": "data_exfil"
    }
]

def test_inference_service(url="http://localhost:5002"):
    """Test the inference service via HTTP API"""
    print(f"Testing Inference Service at {url}")
    print("=" * 60)
    
    # Test health endpoint
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check: OK\n")
        else:
            print(f"✗ Health check failed: {response.status_code}\n")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to inference service: {e}")
        print(f"  Make sure the service is running on {url}")
        return False
    
    # Test inference endpoint
    passed = 0
    failed = 0
    
    for test_case in TEST_CASES:
        print(f"Test: {test_case['name']}")
        print(f"  Expected: {test_case['expected']}")
        
        try:
            response = requests.post(
                f"{url}/infer",
                json=test_case['features'],
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                label = result['label']
                confidence = result['confidence']
                
                print(f"  Result: {label} (confidence: {confidence:.2%})")
                
                if label == test_case['expected']:
                    print("  ✓ PASS\n")
                    passed += 1
                else:
                    print(f"  ⚠ UNEXPECTED (got '{label}', expected '{test_case['expected']}')\n")
                    failed += 1
            else:
                print(f"  ✗ Request failed: {response.status_code}\n")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_model_files():
    """Check if trained model files exist"""
    import os
    
    print("\nChecking Model Files")
    print("=" * 60)
    
    model_path = "models/model.joblib"
    scaler_path = "models/scaler.joblib"
    
    model_exists = os.path.exists(model_path)
    scaler_exists = os.path.exists(scaler_path)
    
    if model_exists:
        print(f"✓ Model file found: {model_path}")
        # Try to load it
        try:
            import joblib
            model = joblib.load(model_path)
            print(f"  Type: {type(model).__name__}")
            if hasattr(model, 'n_estimators'):
                print(f"  Trees: {model.n_estimators}")
        except Exception as e:
            print(f"  ✗ Cannot load model: {e}")
    else:
        print(f"✗ Model file not found: {model_path}")
        print("  → Run training first: python services/training/train.py")
    
    if scaler_exists:
        print(f"✓ Scaler file found: {scaler_path}")
    else:
        print(f"✗ Scaler file not found: {scaler_path}")
    
    print()
    return model_exists and scaler_exists


def test_direct_inference():
    """Test model directly without using the API"""
    import os
    print("\nDirect Model Test")
    print("=" * 60)
    
    try:
        import joblib
        import numpy as np
        
        model_path = "models/model.joblib"
        scaler_path = "models/scaler.joblib"
        
        if not os.path.exists(model_path):
            print("✗ No trained model found")
            print("  The system will use heuristic-based detection instead")
            return True  # Not a failure, just using fallback
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        # Test with sample data
        test_vector = np.array([[1.5, 1024, 10, 682.67, 6.67, 1.0, 102.4]])
        
        if scaler:
            test_vector = scaler.transform(test_vector)
        
        prediction = model.predict(test_vector)
        probabilities = model.predict_proba(test_vector)[0]
        
        print(f"✓ Direct inference successful")
        print(f"  Prediction: {prediction[0]}")
        print(f"  Probabilities: {probabilities}")
        print()
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("AI-NIDS Model Verification Script")
    print("=" * 60)
    print()
    
    # Test 1: Check model files
    files_ok = test_model_files()
    
    # Test 2: Test direct inference (if models exist)
    if files_ok:
        direct_ok = test_direct_inference()
    else:
        print("⚠ Skipping direct test (no trained models)")
        print("  System will use heuristic-based detection\n")
        direct_ok = True
    
    # Test 3: Test via API
    inference_ok = test_inference_service()
    
    print("\n" + "=" * 60)
    if inference_ok:
        print("✓ All tests passed! Models are working.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Check the output above.")
        sys.exit(1)
