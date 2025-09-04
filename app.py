from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load models at startup
temp_model = tf.keras.models.load_model("temperature_model.keras")
hum_model = tf.keras.models.load_model("humidity_model.keras")

@app.route("/")
def home():
    return {"message": "AI Model API running on Railway 🚀"}

@app.route("/predict/temperature", methods=["POST"])
def predict_temp():
    try:
        data = request.json["data"]  # expects list of 24 numbers
        arr = np.array(data, dtype=float).reshape(1, 24, 1)  # reshape for LSTM/GRU
        prediction = temp_model.predict(arr).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/humidity", methods=["POST"])
def predict_hum():
    try:
        data = request.json["data"]
        arr = np.array(data, dtype=float).reshape(1, 24, 1)
        prediction = hum_model.predict(arr).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
