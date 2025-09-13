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
    if score < 30:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

def get_feature_importances(model):
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

def recommend_changes(row, score, level):
    recs = []
    if level == "High":
        if row["recycling_rate"] < 0.5:
            recs.append(f"Increase recycling rate from {row['recycling_rate']*100:.0f}% → 50%+ to lower impact.")
        if row["transport_km"] > 200:
            recs.append(f"Reduce transport distance by {row['transport_km']-200} km → save emissions.")
        if row["smelting_energy_mj_per_kg"] > 70:
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
        top_factors = get_feature_importances(model)[:3]
        recs = recommend_changes(data, score, level)

        # Chart data
        impact_summary = {
            "CO2 (kg/kg)": data["co2_kg_per_kg"],
            "Energy (MJ/kg)": data["smelting_energy_mj_per_kg"] + data["mining_energy_mj_per_kg"],
            "Water (L/kg)": data["water_usage_l_per_kg"]
        }

        circularity_score = round(data["recycling_rate"] * 100, 2)

        scenarios = {
            "Current": score,
            "80% Recycled": model.predict(pd.DataFrame([{**data, "recycling_rate": 0.8}]))[0]
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
    port = int(os.environ.get("PORT", 5000))  # ✅ Use Render's PORT if available
    app.run(host="0.0.0.0", port=port, debug=False)
