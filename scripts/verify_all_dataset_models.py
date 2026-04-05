"""
Verify All Trained Models from Both Datasets
Tests CICIDS2017 Full and UNSW-NB15 models
"""
import os
import sys
import numpy as np
import joblib

def verify_dataset_models(dataset_name, model_dir):
    """Verify all 4 models for a dataset"""
    print(f"\n{'='*70}")
    print(f"Verifying {dataset_name} Models")
    print(f"{'='*70}")
    
    results = {}
    
    # Check for required files
    required_files = {
        'Random Forest': f'{model_dir}/rf_model.joblib',
        'CNN': f'{model_dir}/cnn_model.h5',
        'LSTM': f'{model_dir}/lstm_model.h5',
        'CNN-LSTM': f'{model_dir}/cnn_lstm_model.h5',
        'Scaler': f'{model_dir}/scaler.joblib',
        'Encoder': f'{model_dir}/label_encoder.joblib'
    }
    
    print(f"\n📁 Checking files in {model_dir}/:")
    all_exist = True
    for name, path in required_files.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) / (1024*1024) if exists else 0
        icon = "✅" if exists else "❌"
        print(f"  {icon} {name}: {size:.2f} MB" if exists else f"  {icon} {name}: NOT FOUND")
        if name in ['Random Forest', 'CNN', 'LSTM', 'CNN-LSTM']:
            results[name] = exists
        all_exist = all_exist and exists
    
    if not all_exist:
        print(f"\n❌ {dataset_name}: Missing files!")
        return results
    
    # Load and test each model
    print(f"\n🧪 Testing model loading and inference:")
    
    try:
        # Load preprocessing
        scaler = joblib.load(f'{model_dir}/scaler.joblib')
        encoder = joblib.load(f'{model_dir}/label_encoder.joblib')
        print(f"  ✅ Scaler and encoder loaded")
        print(f"  📊 Classes ({len(encoder.classes_)}): {encoder.classes_[:5]}{'...' if len(encoder.classes_) > 5 else ''}")
        
        # Create test data
        n_features = scaler.n_features_in_
        test_data = np.random.rand(1, n_features)
        test_scaled = scaler.transform(test_data)
        
        # Test Random Forest
        try:
            rf_model = joblib.load(f'{model_dir}/rf_model.joblib')
            rf_pred = rf_model.predict(test_scaled)
            print(f"  ✅ Random Forest: Loaded & working ({rf_model.n_estimators} trees)")
            results['Random Forest'] = True
        except Exception as e:
            print(f"  ❌ Random Forest: Failed - {e}")
            results['Random Forest'] = False
        
        # Test deep learning models
        import tensorflow as tf
        
        for model_name, model_file in [('CNN', 'cnn_model.h5'), 
                                        ('LSTM', 'lstm_model.h5'), 
                                        ('CNN-LSTM', 'cnn_lstm_model.h5')]:
            try:
                model = tf.keras.models.load_model(f'{model_dir}/{model_file}')
                pred = model.predict(test_scaled, verbose=0)
                params = model.count_params()
                print(f"  ✅ {model_name}: Loaded & working ({params:,} params)")
                results[model_name] = True
            except Exception as e:
                print(f"  ❌ {model_name}: Failed - {e}")
                results[model_name] = False
        
        print(f"\n✅ {dataset_name} verification complete!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL VERIFICATION")
    print("="*70)
    print("\nChecking all trained models from both datasets...")
    
    # Verify CICIDS2017 Full
    cicids_results = verify_dataset_models(
        "CICIDS2017 Full (~2.8M samples)",
        "models/cicids_full"
    )
    
    # Verify UNSW-NB15
    unsw_results = verify_dataset_models(
        "UNSW-NB15 (257K samples)",
        "models/unsw_nb15"
    )
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL VERIFICATION SUMMARY")
    print("="*70)
    
    print("\n📊 CICIDS2017 Full Dataset:")
    for model, status in cicids_results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {model}")
    
    print("\n📊 UNSW-NB15 Dataset:")
    for model, status in unsw_results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {model}")
    
    all_cicids = all(cicids_results.values())
    all_unsw = all(unsw_results.values())
    
    print("\n" + "="*70)
    if all_cicids and all_unsw:
        print("🎉 ALL MODELS VERIFIED SUCCESSFULLY!")
        print("="*70)
        print(f"\n✅ 8 models total trained and working:")
        print(f"   • 4 models on CICIDS2017 Full (2.8M samples, 9 attack types)")
        print(f"   • 4 models on UNSW-NB15 (257K samples, 10 attack categories)")
        print(f"\n📁 Model locations:")
        print(f"   • models/cicids_full/")
        print(f"   • models/unsw_nb15/")
        return 0
    else:
        print("⚠️ SOME MODELS FAILED VERIFICATION")
        print("="*70)
        if not all_cicids:
            print(f"\n❌ CICIDS2017 issues detected")
        if not all_unsw:
            print(f"\n❌ UNSW-NB15 issues detected")
        return 1

if __name__ == '__main__':
    sys.exit(main())
