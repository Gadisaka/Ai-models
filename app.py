from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load models once when the server starts
temp_model = tf.keras.models.load_model("temperature_model.keras")
hum_model = tf.keras.models.load_model("humidity_model.keras")

@app.route("/")
def home():
    return {"message": "AI Model API running on Railway 🚀"}

@app.route("/predict/temperature", methods=["POST"])
def predict_temp():
    data = request.json["data"]  # expects list of 24 values
    prediction = temp_model.predict([data]).tolist()
    return jsonify({"prediction": prediction})

@app.route("/predict/humidity", methods=["POST"])
def predict_hum():
    data = request.json["data"]
    prediction = hum_model.predict([data]).tolist()
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
