import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import numpy as np


def load_data(csv_path="heart.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file '{csv_path}' not found. Make sure it is in the same folder."
        )

    df = pd.read_csv(csv_path)

    if "target" not in df.columns:
        raise ValueError("CSV file must contain a 'target' column (0 = no disease, 1 = disease).")

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y, df.columns.tolist()


def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("✅ Model trained successfully!")
    print(f"📊 Accuracy on test set: {accuracy * 100:.2f}%\n")
    print("Detailed classification report:")
    print(classification_report(y_test, y_pred))

    return model, scaler


def save_model(model, scaler, feature_names,
               model_path="heart_model.joblib",
               scaler_path="heart_scaler.joblib",
               features_path="heart_features.txt"):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, "w") as f:
        for name in feature_names:
            if name != "target":
                f.write(name + "\n")

    print(f"\n💾 Model saved as: {model_path}")
    print(f"💾 Scaler saved as: {scaler_path}")
    print(f"💾 Feature names saved as: {features_path}")


def load_model(model_path="heart_model.joblib",
               scaler_path="heart_scaler.joblib",
               features_path="heart_features.txt"):
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Model/scaler/features not found. Train the model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(features_path, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    return model, scaler, feature_names


def get_user_input(feature_names):
    print("\nEnter patient details:")
    values = []
    for feature in feature_names:
        while True:
            try:
                val = float(input(f"  {feature}: "))
                values.append(val)
                break
            except ValueError:
                print("   Please enter a valid number.")
    return values


def predict_one():
    model, scaler, feature_names = load_model()

    user_values = get_user_input(feature_names)
    X_new = np.array(user_values).reshape(1, -1)

    X_new_scaled = scaler.transform(X_new)
    pred = model.predict(X_new_scaled)[0]
    prob = model.predict_proba(X_new_scaled)[0][1]  # probability of disease

    print("\n🔍 Prediction result:")
    if pred == 1:
        print(f"⚠️ The model predicts: Heart Disease PRESENT")
    else:
        print(f"✅ The model predicts: No Heart Disease")

    print(f"Probability of heart disease: {prob * 100:.2f}%")


def main():
    while True:
        print("\n=== Heart Disease Prediction Project ===")
        print("1. Train model from heart.csv")
        print("2. Predict for a new patient (using saved model)")
        print("3. Exit")

        choice = input("Select option (1/2/3): ").strip()

        if choice == "1":
            try:
                X, y, cols = load_data("heart.csv")
                feature_names = [c for c in cols if c != "target"]
                model, scaler = train_model(X, y)
                save_model(model, scaler, feature_names)
            except Exception as e:
                print(f"❌ Error: {e}")

        elif choice == "2":
            try:
                predict_one()
            except Exception as e:
                print(f"❌ Error: {e}")
        elif choice == "3":
            print("👋 Exiting. Bye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
