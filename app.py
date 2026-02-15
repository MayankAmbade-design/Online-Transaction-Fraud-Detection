# ==========================================
# ONLINE TRANSACTION FRAUD DETECTION - FLASK APP
# ==========================================

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

# ------------------------------------------
# Initialize Flask App
# ------------------------------------------

app = Flask(__name__)

# ------------------------------------------
# Load Model
# ------------------------------------------

MODEL_PATH = "fraud_detection_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found!")

model = joblib.load(MODEL_PATH)

EXPECTED_FEATURES = 30   # Time + V1-V28 + Amount


# ------------------------------------------
# Home Route (Web UI)
# ------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------------
# Form Prediction Route
# ------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_input = request.form.get("features")

        if not raw_input:
            return render_template("index.html",
                                   result="❌ Please enter 30 comma-separated values.")

        values = [float(x.strip()) for x in raw_input.split(",") if x.strip()]

        if len(values) != EXPECTED_FEATURES:
            return render_template(
                "index.html",
                result=f"❌ Expected {EXPECTED_FEATURES} values, got {len(values)}"
            )

        features = np.array(values).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            result = f"🚨 Fraudulent Transaction (Probability: {probability:.4f})"
        else:
            result = f"✅ Genuine Transaction (Probability: {probability:.4f})"

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html",
                               result=f"❌ Error: {str(e)}")


# ------------------------------------------
# API Prediction Route (JSON)
# ------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        values = data["features"]

        if len(values) != EXPECTED_FEATURES:
            return jsonify(
                {"error": f"Expected {EXPECTED_FEATURES} features"}
            ), 400

        features = np.array(values).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            "prediction": "Fraudulent" if prediction == 1 else "Genuine",
            "fraud_probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ------------------------------------------
# Run App
# ------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
