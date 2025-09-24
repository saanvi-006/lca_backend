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

def generate_fallback_recommendations(role=None, impact_score=None, impact_level=None, top_features=None):
    """Generate basic recommendations when Gemini API fails."""
    role_prefix = f"As a {role}, " if role else ""
    
    base_recommendations = f"""{role_prefix}here are key sustainability improvements for your manufacturing process:

• **Increase Recycling Rate**: Aim for 70-80% recycled content to significantly reduce environmental impact and resource consumption

• **Optimize Energy Efficiency**: Focus on renewable energy sources and process optimization to reduce carbon footprint

• **Minimize Transport Distance**: Source materials locally when possible to reduce transportation emissions

• **Implement Water Conservation**: Use closed-loop water systems and recycling to minimize water usage

• **Enhance Material Efficiency**: Apply lean manufacturing principles to reduce waste and improve resource utilization

These strategic changes can help move your sustainability score from {impact_level if impact_level else 'current'} to a better category."""

    return base_recommendations

def call_llm(prompt, conversation_id=None, role=None, impact_score=None, impact_level=None, top_features=None):
    """Call Google Gemini API with the given prompt and optional conversation context."""
    try:
        model_instance = genai.GenerativeModel(
            'gemini-1.5-flash',  # Use latest available model
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            }
        )

        # Attach conversation history if available
        if conversation_id and conversation_id in conversation_history:
            context = conversation_history[conversation_id]
            prompt = f"Previous context: {context}\n\n{prompt}"

        logger.info(f"Sending prompt to Gemini API: {prompt[:100]}...")

        response = model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.7,
                top_p=0.8,
                top_k=40
            )
        )

        # Check for blocked responses first
        if response.candidates:
            candidate = response.candidates[0]
            
            # Check finish reason for safety blocks
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                if finish_reason == 2:  # SAFETY
                    logger.warning("Response blocked by safety filters")
                    return generate_fallback_recommendations(role, impact_score, impact_level, top_features)
                elif finish_reason == 3:  # RECITATION
                    logger.warning("Response blocked due to recitation")
                    return generate_fallback_recommendations(role, impact_score, impact_level, top_features)
                elif finish_reason == 4:  # OTHER
                    logger.warning("Response blocked for other reasons")
                    return generate_fallback_recommendations(role, impact_score, impact_level, top_features)

            # Extract text from parts safely
            if candidate.content and candidate.content.parts:
                reply = "".join([
                    part.text for part in candidate.content.parts 
                    if hasattr(part, "text") and part.text
                ])
                if reply.strip():
                    logger.info("Gemini API response received successfully.")
                    return reply.strip()

        # Fallback if no valid response
        logger.warning("Gemini API returned no usable text. Using fallback recommendations.")
        return generate_fallback_recommendations(role, impact_score, impact_level, top_features)

    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}")
        return generate_fallback_recommendations(role, impact_score, impact_level, top_features)

def recommend_changes_top_features(row, model, role, conversation_id):
    """Generate role-specific recommendations using Gemini for top 3 features."""
    top_features = [f for f, _ in get_feature_importances(model)[:3]]
    input_data = {k: row.get(k, 0) for k in top_features}
    impact_score = model.predict(pd.DataFrame([row]))[0]
    impact_level = categorize_level(impact_score)

    # Create more specific and safer prompts
    if role:
        prompt = f"""
        You are a sustainability consultant helping a {role} improve manufacturing processes.
        
        Current situation:
        - Key environmental factors: {', '.join(top_features)}
        - Current values: {input_data}
        - Environmental impact score: {impact_score:.2f} (classified as {impact_level})
        
        Provide 3-5 specific, actionable recommendations for a {role} to improve sustainability.
        Focus on practical steps they can implement. Keep response under 150 words.
        """
    else:
        prompt = f"""
        You are a sustainability consultant for manufacturing processes.
        
        Current situation:
        - Key environmental factors: {', '.join(top_features)}
        - Current values: {input_data}
        - Environmental impact score: {impact_score:.2f} (classified as {impact_level})
        
        Provide 3-5 specific, actionable sustainability improvements.
        Focus on practical implementation steps. Keep response under 150 words.
        """

    recommendations = call_llm(prompt, conversation_id, role, impact_score, impact_level, top_features)

    # Store conversation context
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

        logger.info(f"Processing prediction request for conversation_id: {conversation_id}")

        # Convert inputs to DataFrame
        df_row = pd.DataFrame([inputs])

        # Model prediction
        score = model.predict(df_row)[0]
        level = categorize_level(score)
        top_factors = get_feature_importances(model)[:3]
        
        # Get recommendations with better error handling
        try:
            recs = recommend_changes_top_features(inputs, model, role, conversation_id)
        except Exception as rec_error:
            logger.error(f"Error generating recommendations: {str(rec_error)}")
            recs = generate_fallback_recommendations(role, score, level, [f[0] for f in top_factors])

        # Prepare chart data
        impact_summary = {
            "CO2 (kg/kg)": inputs.get("co2_kg_per_kg", 0),
            "Energy (MJ/kg)": inputs.get("smelting_energy_mj_per_kg", 0) + inputs.get("mining_energy_mj_per_kg", 0),
            "Water (L/kg)": inputs.get("water_usage_l_per_kg", 0)
        }

        circularity_score = round(inputs.get("recycling_rate", 0) * 100, 2)

        # Calculate scenario comparison
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
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/followup", methods=["POST"])
def followup():
    try:
        data = request.get_json()
        conversation_id = data.get("conversation_id")
        question = data.get("question")
        role = data.get("role", "")

        if not conversation_id or conversation_id not in conversation_history:
            logger.error(f"Invalid or missing conversation_id: {conversation_id}")
            return jsonify({"error": "Invalid or missing conversation_id. Please start a new prediction first."}), 400

        if not question or not question.strip():
            return jsonify({"error": "Question cannot be empty"}), 400

        context = conversation_history[conversation_id]
        logger.info(f"Processing follow-up question for conversation_id: {conversation_id}")

        # Create safer follow-up prompts
        if role:
            prompt = f"""
            You are helping a {role} with sustainability questions about their manufacturing process.
            
            Context:
            - Previous inputs: {context['inputs']}
            - Impact score: {context['impact_score']:.2f} ({context['impact_level']})
            - Top factors: {context['top_features']}
            
            Question: {question}
            
            Provide a helpful answer tailored to a {role}'s responsibilities and priorities.
            Keep response under 100 words and be specific and actionable.
            """
        else:
            prompt = f"""
            You are a sustainability consultant answering questions about manufacturing processes.
            
            Context:
            - Previous inputs: {context['inputs']}
            - Impact score: {context['impact_score']:.2f} ({context['impact_level']})
            - Top factors: {context['top_features']}
            
            Question: {question}
            
            Provide a helpful, specific answer. Keep response under 100 words.
            """

        answer = call_llm(prompt, conversation_id, role, context['impact_score'], context['impact_level'], context['top_features'])

        # Update conversation history
        conversation_history[conversation_id]["last_question"] = question
        conversation_history[conversation_id]["last_answer"] = answer

        logger.info(f"Follow-up successful for conversation_id: {conversation_id}")
        return jsonify({"conversation_id": conversation_id, "answer": answer})

    except Exception as e:
        logger.error(f"Follow-up error: {str(e)}")
        return jsonify({"error": f"Follow-up failed: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "LCA API is running"}), 200

# ------------------ Run server ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)

