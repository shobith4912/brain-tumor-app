import io
import base64
import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import cv2

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model("final_unet.h5", compile=False)

IMG_SIZE = 128

# -------------------------------
# SERVE FRONTEND
# -------------------------------
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(img):
    img = img.convert("L").resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)

    # Normalize
    img = img / 255.0
    img_uint8 = (img * 255).astype(np.uint8)

    # Threshold
    _, thresh = cv2.threshold(img_uint8, 30, 255, cv2.THRESH_BINARY)

    # Morph clean
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Largest contour (brain only)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_uint8)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    # Remove skull edges
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Apply mask
    brain = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)

    # Smooth
    brain = cv2.GaussianBlur(brain, (3, 3), 0)

    # Normalize again
    brain = brain / 255.0

    return brain.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# -------------------------------
# POSTPROCESS
# -------------------------------
def postprocess(pred):
    pred_mask = np.argmax(pred[0], axis=-1)
    binary_mask = (pred_mask > 0).astype(np.uint8) * 255

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    return binary_mask

# -------------------------------
# LOCATION
# -------------------------------
def get_location(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return "No tumor"

    cx, cy = int(np.mean(xs)), int(np.mean(ys))

    x_pos = "Left" if cx < IMG_SIZE//3 else "Right" if cx > 2*IMG_SIZE//3 else "Center"
    y_pos = "Top" if cy < IMG_SIZE//3 else "Bottom" if cy > 2*IMG_SIZE//3 else "Middle"

    return f"{y_pos}-{x_pos}"

# -------------------------------
# IMAGE ENCODE
# -------------------------------
def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# -------------------------------
# OVERLAY
# -------------------------------
def create_overlay(original, mask):
    original = original.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    original_np = np.array(original)

    overlay = original_np.copy()
    overlay[mask > 0] = [255, 0, 0]

    blended = (0.7 * original_np + 0.3 * overlay).astype(np.uint8)

    return Image.fromarray(blended)

# -------------------------------
# API ROUTE
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        img = Image.open(file)

        # Inference
        x = preprocess(img)
        pred = model.predict(x)

        mask = postprocess(pred)

        # Stats
        tumor_pixels = int(np.sum(mask > 0))
        total_pixels = IMG_SIZE * IMG_SIZE
        area = round((tumor_pixels / total_pixels) * 100, 2)

        detected = tumor_pixels > 0

        if not detected:
            severity = "None"
        elif area < 5:
            severity = "Low"
        elif area < 15:
            severity = "Moderate"
        else:
            severity = "High"

        confidence = round(float(np.max(pred)) * 100, 2)
        location = get_location(mask)

        # Images
        original_resized = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        mask_img = Image.fromarray(mask)
        overlay_img = create_overlay(img, mask)

        return jsonify({
            "detected": detected,
            "severity": severity,
            "area": area,
            "confidence": confidence,
            "location": location,
            "original_image": encode_image(original_resized),
            "mask_image": encode_image(mask_img),
            "overlay_image": encode_image(overlay_img)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)