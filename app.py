from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return "House Price Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    area = data["area"]
    bedrooms = data["bedrooms"]
    bathrooms = data["bathrooms"]
    age = data["age"]

    features = np.array([[area, bedrooms, bathrooms, age]])
    prediction = model.predict(features)[0]

    return jsonify({"predicted_price": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(debug=True)