from joblib import dump
from sklearn.ensemble import RandomForestClassifier


def train_rf(X_train, y_train, out_path="rf_model.joblib"):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    if out_path:
        dump(model, out_path)
    return model
