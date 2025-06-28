import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("butterfly_model_best.h5")

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
with open("binary_butterfly_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as binary_butterfly_model.tflite")
