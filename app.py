from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

# ------------------ Load trained model ------------------
model = joblib.load("lca_demo_model.joblib")

# ------------------ Helper functions ------------------
def categorize_level(score):
    """Categorize impact score into Low, Medium, High."""
    if score < 30:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

def get_feature_importances(model):
    """Return top contributing features with importance percentages."""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if hasattr(trans, "get_feature_names_out"):
            feature_names.extend(list(trans.get_feature_names_out(cols)))
        else:
            feature_names.extend(cols)

    importances = regressor.feature_importances_
    indices = np.argsort(importances)[::-1]

    return [(feature_names[i], round(importances[i] * 100, 2)) for i in indices]

def recommend_changes_top_features(row, model):
    """Provide recommendations only for the top 3 most important features."""
    recs = []

    top_features = [f for f, _ in get_feature_importances(model)[:3]]

    for f in top_features:
        if f == "recycling_rate" and row.get(f, 0) < 0.5:
            recs.append(f"Increase recycling rate from {row[f]*100:.0f}% → 50%+ to lower impact.")
        elif f == "transport_km" and row.get(f, 0) > 200:
            recs.append(f"Reduce transport distance by {row[f]-200} km → save emissions.")
        elif f == "smelting_energy_mj_per_kg" and row.get(f, 0) > 70:
            recs.append("Optimize smelting energy use (e.g., switch to renewable power).")

    if not recs:
        recs.append("Process is already efficient. Minor improvements possible.")

    return recs

# ------------------ Flask API ------------------
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df_row = pd.DataFrame([data])

        # Model prediction
        score = model.predict(df_row)[0]
        level = categorize_level(score)
        top_factors = get_feature_importances(model)[:3]  # global top 3
        recs = recommend_changes_top_features(data, model)

        # Prepare chart data
        impact_summary = {
            "CO2 (kg/kg)": data.get("co2_kg_per_kg", 0),
            "Energy (MJ/kg)": data.get("smelting_energy_mj_per_kg", 0) + data.get("mining_energy_mj_per_kg", 0),
            "Water (L/kg)": data.get("water_usage_l_per_kg", 0)
        }

        circularity_score = round(data.get("recycling_rate", 0) * 100, 2)

        # Example scenario: 80% recycling
        scenario_data = {**data, "recycling_rate": 0.8}
        scenarios = {
            "Current": float(score),
            "80% Recycled": float(model.predict(pd.DataFrame([scenario_data]))[0])
        }

        # Build final response
        result = {
            "impact_score": float(score),
            "impact_level": level,
            "recommendations": recs,
            "top_factors": dict(top_factors),
            "charts": {
                "impact_summary": impact_summary,
                "circularity_score": circularity_score,
                "scenarios": scenarios
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ------------------ Run server ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
