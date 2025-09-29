from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = joblib.load(f)

@app.route("/")
def home():
    return "ScreenTime vs Mental Wellness Model (Local Flask) is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Convert input to DataFrame
    if isinstance(data, dict) and "instances" in data:
        X_input = pd.DataFrame(data["instances"])
    elif isinstance(data, list):
        X_input = pd.DataFrame(data)
    elif isinstance(data, dict):
        X_input = pd.DataFrame([data])
    else:
        return jsonify({"error": "Invalid input format"}), 400

    # --- Categorical Encoding Handling ---
    X_input = pd.get_dummies(X_input)

    # Ensure all training columns exist
    for col in model.feature_names_in_:
        if col not in X_input.columns:
            X_input[col] = 0

    # Reorder columns to match training
    X_input = X_input[model.feature_names_in_]

    try:
        preds = model.predict(X_input)
        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# from flask import Flask, request, jsonify
# import pandas as pd
# import joblib

# app = Flask(__name__)

# # Load model
# with open("model.pkl", "rb") as f:
#     model = joblib.load(f)

# @app.route("/")
# def home():
#     return "ScreenTime vs Mental Wellness Model (Local Flask) is running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json(force=True)

#     # Convert input to DataFrame
#     if isinstance(data, dict) and "instances" in data:
#         X_input = pd.DataFrame(data["instances"])
#     elif isinstance(data, list):
#         X_input = pd.DataFrame(data)
#     elif isinstance(data, dict):
#         X_input = pd.DataFrame([data])
#     else:
#         return jsonify({"error": "Invalid input format"}), 400

#     # One-hot encode categorical columns
#     X_input = pd.get_dummies(X_input)

#     # Ensure all training columns exist
#     for col in model.feature_names_in_:
#         if col not in X_input.columns:
#             X_input[col] = 0

#     # Reorder columns to match training
#     X_input = X_input[model.feature_names_in_]

#     # Predict
#     try:
#         preds = model.predict(X_input)
#         return jsonify({"predictions": preds.tolist()})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
