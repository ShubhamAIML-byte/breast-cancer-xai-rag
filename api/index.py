# ==============================================================================
# API: api/index.py
# Purpose: Serverless FastAPI backend for Vercel deployment.
#          Provides real-time XGBoost inference, native C++ TreeSHAP attributions,
#          RAG cytopathology guideline retrieval, and LLM natural-language explanations.
# ==============================================================================
import os
import sys
import json
import numpy as np
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb

# Add parent directory to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from api.knowledge_base import ClinicalCytologyRAG

# Initialize FastAPI app
app = FastAPI(
    title="Explainable Breast Cancer Classification & RAG API",
    description="24/7 Serverless XAI Decision Support Backend",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for cached model & metadata
booster = None
feature_names = None
rag_engine = ClinicalCytologyRAG()

def load_model_and_features():
    global booster, feature_names
    if booster is None:
        model_path = os.path.join(BASE_DIR, "model.json")
        if not os.path.exists(model_path):
            raise RuntimeError(f"model.json not found at {model_path}")
        booster = xgb.Booster()
        booster.load_model(model_path)
        
    if feature_names is None:
        features_path = os.path.join(BASE_DIR, "feature_names.json")
        with open(features_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

# Request Models
class PredictRequest(BaseModel):
    features: Dict[str, float]

class ExplainRequest(BaseModel):
    payload: Dict[str, Any]

class ChatRequest(BaseModel):
    payload: Dict[str, Any]
    query: str
    history: Optional[List[Dict[str, str]]] = []

# SYSTEM PROMPT FOR LLM
SYSTEM_PROMPT = """You are an AI explanation assistant for a breast cancer classification research prototype. Explain the supplied machine learning prediction using only the provided XGBoost prediction, feature values, class mapping, SHAP contributions, and retrieved clinical literature from the ChromaDB RAG vector store.
Rules:
1. Clearly state the model's predicted class.
2. Explain the most influential features and whether they pushed the explained model output toward or away from the predicted class.
3. Cite the retrieved clinical pathology guidelines to explain the biological meaning of the top SHAP features.
4. Do not invent feature contributions, clinical findings, probabilities, medical evidence, or causal relationships.
5. Distinguish model explanation from medical diagnosis.
6. Explain that SHAP describes the behavior of the fitted model, not necessarily the biological cause of disease.
7. If the evidence is insufficient, explicitly state that limitation.
8. Use clear, accessible English.
9. Never modify the prediction or calculate SHAP values yourself.
10. Clearly identify the explanation as AI-generated and model-dependent."""

def generate_template_explanation(payload: Dict[str, Any]) -> str:
    pred = payload["model_prediction"]
    pos = payload["top_evidence"]["top_positive_contributors_to_malignancy"]
    neg = payload["top_evidence"]["top_negative_contributors_to_malignancy"]
    rag_lit = payload.get("retrieved_clinical_literature_rag", [])
    
    lines = [
        f"### Diagnostic Summary: Patient Analysis",
        f"**Model Decision**: The XGBoost classifier predicts **{pred['predicted_diagnosis'].upper()}** "
        f"with a model-estimated probability of **{pred['malignant_probability']:.2%}** "
        f"(Confidence: {pred['confidence_percent']:.1f}%).\n",
        f"Baseline log-odds is {pred['base_log_odds']:.2f}, while this sample's final log-odds output is {pred['sample_log_odds']:.2f}.\n",
        "**Key Morphological Drivers (SHAP Attributions):**"
    ]
    if pred["predicted_class_code"] == 1:
        for it in pos[:3]:
            lines.append(f"  • `{it['feature_name']}` = **{it['observed_value']:.3f}** (SHAP: **+{it['shap_value']:.3f}** toward malignancy)")
        if neg:
            lines.append("\nCountervailing factors supporting benign classification:")
            for it in neg[:2]:
                lines.append(f"  • `{it['feature_name']}` = **{it['observed_value']:.3f}** (SHAP: **{it['shap_value']:.3f}** toward benign)")
    else:
        for it in neg[:3]:
            lines.append(f"  • `{it['feature_name']}` = **{it['observed_value']:.3f}** (SHAP: **{it['shap_value']:.3f}** toward benign)")
        if pos:
            lines.append("\nCountervailing factors that exhibited elevated risk:")
            for it in pos[:2]:
                lines.append(f"  • `{it['feature_name']}` = **{it['observed_value']:.3f}** (SHAP: **+{it['shap_value']:.3f}** toward malignancy)")

    if rag_lit:
        lines.append("\n**Retrieved Clinical Pathology Context (RAG Guidelines):**")
        for g in rag_lit:
            lines.append(f"  📖 **[{g['source']}]** {g['text']}")

    lines.append("\n*Clinical Notice: This explanation is generated by an academic AI decision-support research prototype and does not substitute for certified histological evaluation.*")
    return "\n".join(lines)


# ==============================================================================
# ENDPOINTS
# ==============================================================================
@app.get("/")
@app.get("/api")
@app.get("/api/health")
def healthcheck():
    load_model_and_features()
    return {
        "status": "healthy",
        "service": "Explainable Breast Cancer AI (XGBoost + SHAP + RAG + LLM)",
        "features_loaded": len(feature_names),
        "deployment": "24/7 Vercel Serverless"
    }

@app.get("/api/samples")
def get_sample_cases():
    samples_path = os.path.join(BASE_DIR, "data", "test_samples.json")
    if os.path.exists(samples_path):
        with open(samples_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.post("/api/predict")
def predict_and_explain(req: PredictRequest):
    load_model_and_features()
    
    # Build feature vector
    feat_vector = []
    missing_feats = []
    for f in feature_names:
        if f in req.features:
            feat_vector.append(float(req.features[f]))
        else:
            missing_feats.append(f)
            feat_vector.append(0.0)

    # Prepare DMatrix
    X = np.array([feat_vector], dtype=np.float32)
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    
    # 1. Prediction & Margin
    raw_margin = float(booster.predict(dmatrix, output_margin=True)[0])
    prob_malignant = float(1.0 / (1.0 + np.exp(-raw_margin)))
    pred_code = 1 if prob_malignant >= 0.50 else 0
    pred_label = "Malignant" if pred_code == 1 else "Benign"
    
    # 2. Native C++ TreeSHAP Computation
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    phi_values = contribs[:-1]
    base_val = float(contribs[-1])
    
    # Format feature impacts
    impacts = []
    for fn, fv, phi in zip(feature_names, feat_vector, phi_values):
        impacts.append({
            "feature_name": fn,
            "observed_value": round(float(fv), 4),
            "shap_value": round(float(phi), 4),
            "direction": "Pushes Toward Malignant" if phi > 0 else "Pushes Toward Benign"
        })
        
    sorted_impacts = sorted(impacts, key=lambda x: abs(x["shap_value"]), reverse=True)
    top_pos = [f for f in sorted_impacts if f["shap_value"] > 0][:5]
    top_neg = [f for f in sorted_impacts if f["shap_value"] < 0][:5]
    
    # 3. RAG Retrieval for Top Features
    top_driver_names = [f["feature_name"] for f in sorted_impacts[:3]]
    retrieved_lit = rag_engine.retrieve_guidelines(top_driver_names, top_k=2)
    
    # Assemble structured evidence payload
    payload = {
        "metadata": {
            "model": "Tuned XGBoost Classifier",
            "explainer": "Exact C++ TreeSHAP (Additive Log-Odds Margin)",
            "rag": "Clinical Breast Cytopathology Vector Store (WHO & IAC Yokohama)"
        },
        "model_prediction": {
            "predicted_class_code": pred_code,
            "predicted_diagnosis": pred_label,
            "malignant_probability": round(prob_malignant, 4),
            "confidence_percent": round(max(prob_malignant, 1 - prob_malignant) * 100, 2),
            "base_log_odds": round(base_val, 4),
            "sample_log_odds": round(raw_margin, 4)
        },
        "top_evidence": {
            "top_positive_contributors_to_malignancy": top_pos,
            "top_negative_contributors_to_malignancy": top_neg,
            "all_top_features_ranked": sorted_impacts[:10]
        },
        "retrieved_clinical_literature_rag": retrieved_lit
    }
    
    # Generate default text explanation
    payload["explanation"] = generate_template_explanation(payload)
    return payload

@app.post("/api/explain")
def generate_llm_explanation(req: ExplainRequest):
    payload = req.payload
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not api_key:
        return {"explanation": generate_template_explanation(payload), "mode": "Deterministic Fallback (No API Key)"}
        
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a comprehensive, faithful explanation using the payload:\n{json.dumps(payload, indent=2)}"}
            ],
            temperature=0.0,
            max_tokens=550
        )
        return {"explanation": res.choices[0].message.content, "mode": "OpenAI GPT-4o-mini"}
    except Exception as e:
        return {"explanation": generate_template_explanation(payload), "mode": f"Fallback (API Error: {str(e)})"}

@app.post("/api/chat")
def chat_followup(req: ChatRequest):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    payload = req.payload
    query = req.query
    
    if not api_key:
        pred = payload["model_prediction"]
        top_f = [f["feature_name"] for f in payload["top_evidence"]["all_top_features_ranked"][:3]]
        fallback_msg = (
            f"Based on the prediction payload for this case: The model classified the sample as "
            f"**{pred['predicted_diagnosis']}** with {pred['malignant_probability']:.2%} probability. "
            f"The primary morphological features influencing this decision were {', '.join(top_f)}. "
            f"To enable live conversational Q&A, provide an OPENAI_API_KEY."
        )
        return {"reply": fallback_msg}
        
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"CURRENT PATIENT EVIDENCE PAYLOAD:\n{json.dumps(payload, indent=2)}"}
        ]
        for turn in req.history[-6:]:
            messages.append(turn)
        messages.append({"role": "user", "content": query})
        
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=400
        )
        return {"reply": res.choices[0].message.content}
    except Exception as e:
        return {"reply": f"Error communicating with AI model: {str(e)}"}
