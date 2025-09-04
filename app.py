from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load models at startup
temp_model = tf.keras.models.load_model("temperature_model.keras")
hum_model = tf.keras.models.load_model("humidity_model.keras")

# Min/Max for denormalization
MIN_TEMP, MAX_TEMP = 3.0, 90.0
MIN_HUM, MAX_HUM = 4.0, 243.0

def denorm(x, min_v, max_v):
    return x * (max_v - min_v) + min_v

@app.route("/")
def home():
    return {"message": "AI Model API running on Railway 🚀"}

@app.route("/predict/temperature", methods=["POST"])
def predict_temp():
    try:
        data = request.json["data"]  # expects list of 24 numbers
        arr = np.array(data, dtype=float).reshape(1, 24, 1)  # reshape for LSTM
        pred = temp_model.predict(arr).tolist()  # [[scaled]]
        scaled = float(pred[0][0])
        return jsonify({
            "scaled": scaled,
            "denorm": denorm(scaled, MIN_TEMP, MAX_TEMP)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/humidity", methods=["POST"])
def predict_hum():
    try:
        data = request.json["data"]
        arr = np.array(data, dtype=float).reshape(1, 24, 1)
        pred = hum_model.predict(arr).tolist()
        scaled = float(pred[0][0])
        return jsonify({
            "scaled": scaled,
            "denorm": denorm(scaled, MIN_HUM, MAX_HUM)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
