"""Check what the trained Random Forest model expects"""

import joblib
import numpy as np

# Load the model
model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')

print("\n" + "="*70)
print("RANDOM FOREST MODEL INSPECTION")
print("="*70)

print(f"\nModel type: {type(model)}")
print(f"Number of features expected: {model.n_features_in_}")
print(f"\nModel classes: {model.classes_}")
print(f"Number of trees: {model.n_estimators}")

print(f"\n\nScaler:")
print(f"Number of features: {scaler.n_features_in_}")
print(f"Feature means shape: {scaler.mean_.shape}")

# Try to get feature names if available
if hasattr(model, 'feature_names_in_'):
    print(f"\nFeature names from model:")
    for i, name in enumerate(model.feature_names_in_, 1):
        print(f"  {i:2d}. {name}")
else:
    print("\n⚠️  Model doesn't have feature_names_in_ attribute")
    print(f"   Model expects {model.n_features_in_} features (no names stored)")

print("\n" + "="*70)
