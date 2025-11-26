from flask import Flask, render_template, request
import numpy as np
import joblib
import os

MODEL_PATH = "heart_model.joblib"
SCALER_PATH = "heart_scaler.joblib"
FEATURES_PATH = "heart_features.txt"

app = Flask(__name__)


def load_model_and_features():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH)):
        raise FileNotFoundError(
            "Model/scaler/features not found. "
            "Run your training script (option 1) first to create them."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURES_PATH, "r") as f:
        feature_names = [line.strip() for line in f.readlines() if line.strip()]

    return model, scaler, feature_names


# Load once when the app starts
model, scaler, feature_names = load_model_and_features()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    error = None
    input_values = {}

    if request.method == "POST":
        values = []
        try:
            for feat in feature_names:
                val_str = request.form.get(feat, "").strip()
                input_values[feat] = val_str

                if val_str == "":
                    raise ValueError(f"Please enter a value for '{feat}'.")

                values.append(float(val_str))

            X = np.array(values).reshape(1, -1)
            X_scaled = scaler.transform(X)

            pred = int(model.predict(X_scaled)[0])
            prob = float(model.predict_proba(X_scaled)[0][1])

            result = "Heart Disease PRESENT" if pred == 1 else "No Heart Disease"
            probability = round(prob * 100, 2)

        except Exception as e:
            error = str(e)

    return render_template(
        "form.html",
        feature_names=feature_names,
        result=result,
        probability=probability,
        error=error,
        input_values=input_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
