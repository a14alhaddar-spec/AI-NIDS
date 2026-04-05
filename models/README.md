Place trained models here (e.g., model.joblib, scaler.joblib).

Model builders:
- [models/cnn.py](models/cnn.py) -> build_cnn(input_shape, classes)
- [models/lstm.py](models/lstm.py) -> build_lstm(input_shape, classes)
- [models/cnn_lstm.py](models/cnn_lstm.py) -> build_cnn_lstm(input_shape, classes)
- [models/random_forest.py](models/random_forest.py) -> train_rf(X_train, y_train)

Expected input: fixed-length feature vectors (default from preprocessing/features.yml).
