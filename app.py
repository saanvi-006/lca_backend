from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import logging

# ------------------ Setup logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Load trained model ------------------
model = joblib.load("lca_demo_model.joblib")

# ------------------ Gemini API Configuration ------------------
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# In-memory conversation store (for simplicity; use a database for production)
conversation_history = {}

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

def call_llm(prompt, conversation_id=None):
    """Call Google Gemini API with the given prompt and optional conversation context."""
    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            }
        )
        if conversation_id and conversation_id in conversation_history:
            prompt = f"Previous context: {conversation_history[conversation_id]}\n\n{prompt}"
        logger.info(f"Sending prompt to Gemini API: {prompt[:100]}...")  # Log first 100 chars
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500
            )
        )
        if response.candidates and response.candidates[0].finish_reason == 2:  # SAFETY
            logger.warning("Gemini API blocked response due to safety filters.")
            return "Response blocked by safety filters. Try simplifying the request."
        if not response.text:
            logger.warning("Gemini API returned no valid response text.")
            return "No valid response from Gemini API."
        logger.info("Gemini API response received successfully.")
        return response.text
    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}")
        return f"Gemini API Error: {str(e)}"

def recommend_changes_top_features(row, model, role, conversation_id):
    """Generate role-specific recommendations using Gemini for top 3 features."""
    top_features = [f for f, _ in get_feature_importances(model)[:3]]
    input_data = {k: row.get(k, 0) for k in top_features}
    impact_score = model.predict(pd.DataFrame([row]))[0]
    impact_level = categorize_level(impact_score)

    if role:
        prompt = f"""
        As an expert in sustainable manufacturing, assist a {role}.
        Given key factors: {top_features} with values {input_data} and score {impact_score:.2f} ({impact_level}).
        Suggest practical sustainability improvements tailored to a {role}'s role, in under 150 words.
        """
    else:
        prompt = f"""
        As an expert in sustainable manufacturing, suggest practical sustainability improvements.
        Given key factors: {top_features} with values {input_data} and score {impact_score:.2f} ({impact_level}).
        Keep it concise, under 150 words.
        """

    recommendations = call_llm(prompt, conversation_id)

    if conversation_id:
        conversation_history[conversation_id] = {
            "inputs": row,
            "impact_score": impact_score,
            "impact_level": impact_level,
            "top_features": top_features,
            "recommendations": recommendations,
            "role": role
        }

    return recommendations

# ------------------ Flask API ------------------
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        role = data.get("role", "")
        inputs = data.get("inputs", {})
        conversation_id = data.get("conversation_id", str(np.random.randint(1000000)))

        # Convert inputs to DataFrame
        df_row = pd.DataFrame([inputs])

        # Model prediction
        score = model.predict(df_row)[0]
        level = categorize_level(score)
        top_factors = get_feature_importances(model)[:3]
        recs = recommend_changes_top_features(inputs, model, role, conversation_id)

        # Prepare chart data
        impact_summary = {
            "CO2 (kg/kg)": inputs.get("co2_kg_per_kg", 0),
            "Energy (MJ/kg)": inputs.get("smelting_energy_mj_per_kg", 0) + inputs.get("mining_energy_mj_per_kg", 0),
            "Water (L/kg)": inputs.get("water_usage_l_per_kg", 0)
        }

        circularity_score = round(inputs.get("recycling_rate", 0) * 100, 2)

        scenario_data = {**inputs, "recycling_rate": 0.8}
        scenarios = {
            "Current": float(score),
            "80% Recycled": float(model.predict(pd.DataFrame([scenario_data]))[0])
        }

        result = {
            "conversation_id": conversation_id,
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

        logger.info(f"Prediction successful for conversation_id: {conversation_id}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/followup", methods=["POST"])
def followup():
    try:
        data = request.get_json()
        conversation_id = data.get("conversation_id")
        question = data.get("question")
        role = data.get("role", "")

        if not conversation_id or conversation_id not in conversation_history:
            logger.error(f"Invalid or missing conversation_id: {conversation_id}")
            return jsonify({"error": "Invalid or missing conversation_id"}), 400

        context = conversation_history[conversation_id]

        if role:
            prompt = f"""
            As an expert in sustainable manufacturing, assist a {role}.
            Given previous inputs: {context['inputs']} and score: {context['impact_score']:.2f} ({context['impact_level']}).
            Answer this question: {question}
            Keep the response concise, under 100 words, tailored to a {role}'s priorities.
            """
        else:
            prompt = f"""
            As an expert in sustainable manufacturing, answer this question: {question}
            Given previous inputs: {context['inputs']} and score: {context['impact_score']:.2f} ({context['impact_level']}).
            Keep the response concise, under 100 words.
            """

        answer = call_llm(prompt, conversation_id)

        conversation_history[conversation_id]["last_question"] = question
        conversation_history[conversation_id]["last_answer"] = answer

        logger.info(f"Follow-up successful for conversation_id: {conversation_id}")
        return jsonify({"conversation_id": conversation_id, "answer": answer})

    except Exception as e:
        logger.error(f"Follow-up error: {str(e)}")
        return jsonify({"error": str(e)}), 400

# ------------------ Run server ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
