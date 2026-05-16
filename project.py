import numpy as np
# tensorflow packages
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

train_data, test_data = tfds.load(
    name="imdb_reviews",
    split=["train", "test"],
    as_supervised=True
)

model = tf.keras.Sequential([
    hub.KerasLayer(
        "https://tfhub.dev/google/nnlm-en-dim50/2",
        input_shape=[],
        dtype=tf.string,
        trainable=True
    ),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data.batch(512), epochs=5)

results = model.evaluate(test_data.batch(512))

print(results)
