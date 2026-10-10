# Explainable Breast Cancer Decision Support (XGBoost + SHAP + ChromaDB RAG + LLM)

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Deploy_24%2F7-black?style=flat&logo=vercel)](https://vercel.com/)
[![Google Colab](https://img.shields.io/badge/Google_Colab-Open_Notebook-orange?style=flat&logo=googlecolab)](https://colab.research.google.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end clinical Explainable AI (XAI) and Generative AI research application for breast cancer fine-needle aspiration (FNA) diagnostic decision support.

---

## 🔬 System Architecture

```
[Patient Fine-Needle Biopsy Sample]
               │
               ▼
      [Tuned XGBoost Model] ───> Predicted Probability (e.g., 98.4% Malignant)
               │
               ▼
     [SHAP TreeExplainer]   ───> Identifies Top Driver Features:
                                 ['worst_perimeter', 'mean_concave_points']
                                       │
                                       ▼ (Vector Semantic Query)
               ┌─────────────────────────────────────────────────────────┐
               │              ChromaDB RAG Vector Database               │
               │   Indexes Peer-Reviewed Clinical Cytopathology Rules:   │
               │   • WHO Classification of Breast Tumours                │
               │   • IAC Yokohama System for Reporting Breast FNA        │
               │   • Nuclear Pleomorphism & Contour Atypia Standards     │
               └───────────────────────────┬─────────────────────────────┘
                                           │
                                           ▼ (Top Relevant Clinical Guidelines)
┌────────────────────────────────────────────────────────────────────────┐
│                      Enriched Evidence Payload                         │
│  1. Machine Learning Prediction: Malignant (p = 98.42%)                │
│  2. SHAP Mathematical Attribution: worst perimeter (+2.41 log-odds)   │
│  3. Retrieved ChromaDB Guidelines: "Nuclear membrane grooving and      │
│     concave points indicate irregular lamina assembly & aneuploidy,    │
│     characteristic of IAC Yokohama Category 5 ductal carcinoma."       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
          [Generative AI / LLM Engine (GPT-4o-mini / Fallback)]
                                   │
                                   ▼
        [Doctor-Grade Explanation Grounded in Math AND Medical Literature]
```

---

## 🚀 Live 24/7 Deployment on Vercel

This repository is pre-configured with `vercel.json` and a serverless FastAPI backend (`api/index.py`) with a responsive UI (`public/index.html`).

### Step 1: Push this Repository to GitHub
In your terminal, run:
```bash
git init
git add .
git commit -m "Initial commit: Explainable Breast Cancer Decision Support"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
git push -u origin main
```

### Step 2: Deploy to Vercel (1-Click)
1. Go to **[vercel.com](https://vercel.com/)** and log in with your GitHub account.
2. Click **"Add New"** $\rightarrow$ **"Project"**.
3. Select your newly pushed GitHub repository and click **"Import"**.
4. *(Optional)* Under **Environment Variables**, add:
   * **Key**: `OPENAI_API_KEY`
   * **Value**: `your-openai-api-key`
   *(If not provided, the app uses the built-in deterministic fallback engine).*
5. Click **"Deploy"**.

**Your app will be live 24/7 with a public `.vercel.app` URL!** Any future commit you push to GitHub will automatically trigger continuous deployment.

---

## 💻 Local Development

To run the application locally on your computer:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run local FastAPI / Uvicorn server
uvicorn api.index:app --reload --port 8000
```
Open `http://localhost:8000` in your browser.

---

## 📓 Running in Google Colab

To run the complete interactive research notebook:
1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Upload `Explainable_Breast_Cancer_Classification_XGBoost_SHAP_LLM.ipynb`.
3. Click `Runtime` $\rightarrow$ `Run all`.

---

## 📊 Evaluation & Research Benchmark Results

| Model | Accuracy | Malignant Sensitivity | Specificity | F1-Score | ROC-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | 0.9647 | 0.9062 | 0.9811 | 0.9508 | 0.9965 | 0.9261 |
| **Random Forest** | 0.9529 | 0.8750 | 0.9811 | 0.9333 | 0.9941 | 0.9020 |
| **XGBoost (Tuned)** | **0.9767** | **0.9375** | **1.0000** | **0.9677** | **1.0000** | **0.9508** |

*95% Stratified Bootstrap Confidence Intervals ($B = 1000$):*
* **Malignant Recall**: $[0.8438, 1.0000]$
* **ROC-AUC**: $[1.0000, 1.0000]$
* **Accuracy**: $[0.9419, 1.0000]$
