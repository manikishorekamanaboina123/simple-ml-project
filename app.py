from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return """
    <h2>House Price Prediction</h2>
    <form action="/predict-form" method="post">
        <label>Area:</label><br>
        <input type="number" name="area"><br><br>

        <label>Bedrooms:</label><br>
        <input type="number" name="bedrooms"><br><br>

        <label>Bathrooms:</label><br>
        <input type="number" name="bathrooms"><br><br>

        <label>Age:</label><br>
        <input type="number" name="age"><br><br>

        <button type="submit">Predict Price</button>
    </form>
    """

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

@app.route("/predict-form", methods=["POST"])
def predict_form():
    area = float(request.form["area"])
    bedrooms = float(request.form["bedrooms"])
    bathrooms = float(request.form["bathrooms"])
    age = float(request.form["age"])

    features = np.array([[area, bedrooms, bathrooms, age]])
    prediction = model.predict(features)[0]

    return f"<h3>Predicted Price: ${round(float(prediction), 2)}</h3><br><a href='/'>Try again</a>"

if __name__ == "__main__":
    app.run(debug=True)