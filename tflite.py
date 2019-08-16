# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 20:53:31 2019

@author: joyna
"""

# Generate tf.keras model.
import tensorflow as tf
import numpy as np

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=5, input_shape=[1]))
model.add(tf.keras.layers.Dense(units=1))

model.summary()

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=50)

print(model.predict([10.0]))

# Save tf.keras model in HDF5 format.
keras_file = "keras_model.h5"
tf.keras.models.save_model(model, keras_file)

# Convert to TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)