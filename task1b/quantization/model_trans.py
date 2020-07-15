#### For quantizing the keras model to TF lite model

import tensorflow as tf

# Dynamic range quantization
model = tf.keras.models.load_model("put-the-ori-trained-keras-model-here.hdf5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the converted model to TF lite file
with open("name-the-quantized-model-here.tflite", "wb") as output_file:
    output_file.write(tflite_quant_model)
