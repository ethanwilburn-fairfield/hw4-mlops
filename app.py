from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

MODEL_PATH = "model/model.pkl"
SCHEMA_PATH = "model/schema.json"

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

REQUIRED_FEATURES = schema["required_features"]
NUMERIC_FEATURES = schema["numeric_features"]
CATEGORICAL_FEATURES = schema["categorical_features"]
ALLOWED_CATEGORIES = schema["allowed_categories"]


def validate_record(record):
    errors = {}

    missing = [col for col in REQUIRED_FEATURES if col not in record]
    if missing:
        errors["missing_fields"] = missing
        return errors

    for col in NUMERIC_FEATURES:
        value = record.get(col)

        try:
            value = float(value)
        except (TypeError, ValueError):
            errors[col] = "must be a numeric value"
            continue

        if col in ["total_item_price", "total_freight_value", "total_order_value", "freight_ratio"]:
            if value < 0:
                errors[col] = "must be a non-negative number"

        if col in ["num_items", "num_sellers", "order_complexity"]:
            if value < 0:
                errors[col] = "must be a non-negative number"

        if col == "order_hour" and not (0 <= value <= 23):
            errors[col] = "must be between 0 and 23"

        if col == "order_dayofweek" and not (0 <= value <= 6):
            errors[col] = "must be between 0 and 6"

        if col == "is_late" and value not in [0, 1]:
            errors[col] = "must be 0 or 1"

    for col in CATEGORICAL_FEATURES:
        value = str(record.get(col))
        if value not in ALLOWED_CATEGORIES[col]:
            errors[col] = f"unrecognized value '{value}'"

    return errors


def prepare_dataframe(records):
    df = pd.DataFrame(records)
    df = df[REQUIRED_FEATURES].copy()

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    return df


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "loaded"})


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid input", "details": "Request body must be valid JSON"}), 400

    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid input", "details": "Request body must be a JSON object"}), 400

    errors = validate_record(data)
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    df = prepare_dataframe([data])

    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0, 1])
    label = "positive" if pred == 1 else "negative"

    return jsonify({
        "prediction": pred,
        "probability": round(prob, 4),
        "label": label
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if not request.is_json:
        return jsonify({"error": "Invalid input", "details": "Request body must be valid JSON"}), 400

    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({"error": "Invalid input", "details": "Request body must be a JSON array of records"}), 400

    if len(data) == 0:
        return jsonify({"error": "Invalid input", "details": "Batch request cannot be empty"}), 400

    if len(data) > 100:
        return jsonify({"error": "Invalid input", "details": "Batch size cannot exceed 100 records"}), 400

    batch_errors = {}
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            batch_errors[f"record_{i}"] = "must be a JSON object"
            continue

        record_errors = validate_record(record)
        if record_errors:
            batch_errors[f"record_{i}"] = record_errors

    if batch_errors:
        return jsonify({"error": "Invalid input", "details": batch_errors}), 400

    df = prepare_dataframe(data)

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    results = []
    for pred, prob in zip(preds, probs):
        pred = int(pred)
        results.append({
            "prediction": pred,
            "probability": round(float(prob), 4),
            "label": "positive" if pred == 1 else "negative"
        })

    return jsonify({
        "count": len(results),
        "predictions": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
