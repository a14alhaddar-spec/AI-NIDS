import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_shape, classes, learning_rate=1e-3):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Reshape((input_shape, 1))(inputs)
    x = layers.Conv1D(64, 2, activation="relu")(x)
    x = layers.Conv1D(128, 2, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
