import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)  

model = joblib.load("random_forest_model.pkl")

@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.get_json()

    features = [
        data["System Size (kW)"],
        data["Solar_Irradiance"],
        data["Temperature_2m"],
        data["Cloud_Cover"]
    ]

    prediction = model.predict([features])
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
