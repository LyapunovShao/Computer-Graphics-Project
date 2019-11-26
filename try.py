import tensorflow as tf
from layers.sift import *

"""
inputs = [
    tf.keras.Input(shape=(8192, 3), batch_size=16),
    tf.keras.Input(shape=(1024, 3), batch_size=16),
    tf.keras.Input(shape=()),
    tf.keras.Input(shape=(1024, 256), batch_size=16)
]
outputs = pointnet_fp_module([128, 128, 128])(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
"""
inputs = [tf.keras.Input(shape=(8192, 3), batch_size=16), tf.keras.Input(shape=())]
outputs = pointSIFT_res_module(0.1, 64, merge='concat')(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
"""
inputs = [tf.keras.Input(shape=(1024, 3), batch_size=16), tf.keras.Input(shape=(1024, 256), batch_size=16)]
outputs = pointSIFT_module(0.5, 512)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
x = [tf.ones((16, 8192, 3)), tf.ones((16, 8192, 64))]
"""
"""
encoder_input = tf.keras.Input(shape=(28, 28, 1), name='img')
x = tf.keras.layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(3)(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
x = tf.keras.layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = tf.keras.layers.GlobalMaxPooling2D()(x)

encoder = tf.keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()
"""
