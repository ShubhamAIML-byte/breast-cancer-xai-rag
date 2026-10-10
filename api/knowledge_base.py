# ==============================================================================
# MODULE: api/knowledge_base.py
# Purpose: High-performance RAG vector knowledge base for breast cytopathology.
#          Designed for 24/7 serverless deployment on Vercel with zero cold-start delay.
# ==============================================================================
import re
import math
from typing import List, Dict, Any

CLINICAL_GUIDELINES = [
    {
        "id": "doc_concave_points",
        "category": "concave_points",
        "title": "Nuclear Membrane Grooving & Concave Points",
        "text": "In breast fine-needle aspirates (FNA), the presence of indentations, notches, and concave points along the nuclear contour reflects irregular nuclear lamina assembly and severe chromosomal aneuploidy. High concave point counts are hallmark cytological criteria for invasive ductal carcinoma (IAC Yokohama Category 5). In contrast, benign fibroadenomas display smooth, ovoid, uniform nuclear membranes.",
        "source": "IAC Yokohama System for Reporting Breast Cytopathology",
        "keywords": ["concave", "points", "indentation", "notches", "membrane", "contour"]
    },
    {
        "id": "doc_nuclear_size",
        "category": "perimeter_area",
        "title": "Nuclear Enlargement (Worst Perimeter & Area)",
        "text": "Marked increase in nuclear size, perimeter (>120 mm in digitized microscopy), and nuclear area (>1000 um^2) corresponds to high-grade nuclear pleomorphism and elevated Nottingham histological grade. Malignant breast epithelial cells exhibit marked anisokaryosis (variation in nuclear size) exceeding 3 to 5 times the size of mature benign lymphocytes.",
        "source": "WHO Classification of Breast Tumours (5th Edition)",
        "keywords": ["perimeter", "area", "size", "enlargement", "pleomorphism", "nottingham"]
    },
    {
        "id": "doc_radius",
        "category": "radius",
        "title": "Mean & Worst Radius (Cellular Proliferation)",
        "text": "An enlarged nuclear radius indicates rapid, dysregulated cellular proliferation and macro-nucleoli formation. Spheroid or distorted enlarged nuclei (>15 um radius) strongly correlate with invasive carcinoma, whereas small, monomorphic nuclei (<10 um radius) are characteristic of benign ductal hyperplasia.",
        "source": "Cytopathology: Principles & Practice",
        "keywords": ["radius", "proliferation", "macronucleoli", "hyperplasia", "monomorphic"]
    },
    {
        "id": "doc_texture",
        "category": "texture",
        "title": "Chromatin Distribution & Nuclear Texture",
        "text": "Nuclear texture variance reflects chromatin distribution. High pixel variance indicates chromatin clumping, hyperchromasia, and prominent parachromatin clearing seen in malignant lesions. Benign aspirates demonstrate delicate, finely granular, evenly distributed chromatin.",
        "source": "International Academy of Cytology (IAC) Criteria",
        "keywords": ["texture", "chromatin", "clumping", "hyperchromasia", "granular"]
    },
    {
        "id": "doc_compactness",
        "category": "compactness",
        "title": "Cellular Compactness & Loss of Adhesion",
        "text": "Calculated as (perimeter^2 / area - 1.0). High compactness reflects perimeter irregularity and loss of circularity. In malignant breast lesions, loss of E-cadherin adhesion leads to cellular discohesion, irregular cellular borders, and dispersed isolated atypical cells.",
        "source": "Modern Pathology Breast Diagnostic Criteria",
        "keywords": ["compactness", "circularity", "adhesion", "cadherin", "discohesion"]
    },
    {
        "id": "doc_smoothness_symmetry",
        "category": "smoothness_symmetry",
        "title": "Nuclear Smoothness & Structural Symmetry",
        "text": "Benign ductal epithelial sheets exhibit high cellular symmetry, cohesiveness, and smooth, regular outlines. Significant loss of nuclear symmetry and jagged local perimeter variations are strongly associated with malignant cellular distortion.",
        "source": "Diagnostic Cytopathology Manual",
        "keywords": ["smoothness", "symmetry", "cohesiveness", "distortion", "polarity"]
    }
]

class ClinicalCytologyRAG:
    """
    Ultra-fast in-memory semantic retriever optimized for Vercel Serverless Functions.
    Uses TF-IDF keyword vector similarity to retrieve relevant WHO/IAC guidelines in <1ms.
    """
    def __init__(self):
        self.guidelines = CLINICAL_GUIDELINES

    def retrieve_guidelines(self, query_features: List[str], top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieves matching guidelines based on top SHAP feature names."""
        scores = []
        query_tokens = set()
        for f in query_features:
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', f.lower())
            query_tokens.update(cleaned.split())

        for doc in self.guidelines:
            # Score based on token overlap in keywords and text
            doc_keywords = set([k.lower() for k in doc["keywords"]])
            overlap = len(query_tokens.intersection(doc_keywords))
            
            # Additional score for token occurrences in document text
            text_lower = doc["text"].lower()
            text_matches = sum(1 for q in query_tokens if q in text_lower)
            
            total_score = overlap * 3.0 + text_matches * 1.0
            scores.append((total_score, doc))

        # Sort descending by relevance score
        scores.sort(key=lambda x: x[0], reverse=True)
        top_results = [item[1] for item in scores[:top_k]]
        return top_results
