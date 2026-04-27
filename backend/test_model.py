import tensorflow as tf

try:
    model = tf.keras.models.load_model("best_unet.h5", compile=False)
    print("✅ Model loaded successfully!")

    model.summary()

except Exception as e:
    print("❌ Error loading model:")
    print(e)