"""
Verify All Trained Models
Tests that all 4 models can be loaded and make predictions
"""
import os
import sys
import numpy as np

def test_model(model_name, model_path, scaler_path, encoder_path=None, is_dl=False):
    """Test a single model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} Model")
    print(f"{'='*60}")
    
    # Check files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file not found: {scaler_path}")
        return False
    
    print(f"✓ Model file exists: {model_path}")
    print(f"✓ Scaler file exists: {scaler_path}")
    
    if encoder_path and not os.path.exists(encoder_path):
        print(f"❌ Encoder file not found: {encoder_path}")
        return False
    if encoder_path:
        print(f"✓ Encoder file exists: {encoder_path}")
    
    # Load model
    try:
        import joblib
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded successfully")
        
        if encoder_path:
            encoder = joblib.load(encoder_path)
            print(f"✓ Encoder loaded successfully")
            print(f"  Classes: {encoder.classes_}")
        
        if is_dl:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print(f"✓ Deep learning model loaded successfully")
            print(f"  Parameters: {model.count_params():,}")
        else:
            model = joblib.load(model_path)
            print(f"✓ Traditional ML model loaded successfully")
            if hasattr(model, 'n_estimators'):
                print(f"  Trees: {model.n_estimators}")
        
        # Test prediction
        test_data = np.array([[80, 5000, 10, 8, 500, 400, 1000]])
        test_scaled = scaler.transform(test_data)
        
        if is_dl:
            pred = model.predict(test_scaled, verbose=0)
            pred_class = np.argmax(pred[0])
            if encoder_path:
                label = encoder.inverse_transform([pred_class])[0]
                confidence = pred[0][pred_class]
                print(f"✓ Test prediction successful: {label} (confidence: {confidence:.2%})")
            else:
                print(f"✓ Test prediction successful: class {pred_class}")
        else:
            pred = model.predict(test_scaled)
            print(f"✓ Test prediction successful: {pred[0]}")
        
        print(f"\n✅ {model_name} model is WORKING")
        return True
        
    except Exception as e:
        print(f"❌ Error loading/testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("AI-NIDS Model Verification")
    print("="*60)
    
    results = {}
    
    # Test Random Forest
    results['Random Forest'] = test_model(
        'Random Forest',
        'models/model.joblib',
        'models/scaler.joblib',
        is_dl=False
    )
    
    # Test CNN
    results['CNN'] = test_model(
        'CNN',
        'models/cnn_model.h5',
        'models/cnn_scaler.joblib',
        'models/cnn_encoder.joblib',
        is_dl=True
    )
    
    # Test LSTM
    results['LSTM'] = test_model(
        'LSTM',
        'models/lstm_model.h5',
        'models/lstm_scaler.joblib',
        'models/lstm_encoder.joblib',
        is_dl=True
    )
    
    # Test CNN-LSTM
    results['CNN-LSTM'] = test_model(
        'CNN-LSTM',
        'models/cnn_lstm_model.h5',
        'models/cnn_lstm_scaler.joblib',
        'models/cnn_lstm_encoder.joblib',
        is_dl=True
    )
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    for model_name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {model_name}: {'PASSED' if status else 'FAILED'}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL MODELS VERIFIED SUCCESSFULLY!")
        print("="*60)
        print("\nYou can now run all dashboards with:")
        print("  .\\START_ALL_DASHBOARDS.bat")
        print("\nDashboard URLs:")
        print("  Random Forest: http://localhost:8081")
        print("  CNN-LSTM:      http://localhost:8082")
        print("  CNN:           http://localhost:8083")
        print("  LSTM:          http://localhost:8084")
    else:
        print("⚠️  SOME MODELS FAILED VERIFICATION")
        print("="*60)
        failed = [name for name, status in results.items() if not status]
        print(f"\nFailed models: {', '.join(failed)}")
    print()
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
