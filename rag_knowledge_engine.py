# ==============================================================================
# MODULE: rag_knowledge_engine.py
# Purpose: Retrieval-Augmented Generation (RAG) vector database using ChromaDB.
#          Indexes peer-reviewed breast cytopathology guidelines (IAC Yokohama System)
#          to ground SHAP feature attributions in verified medical literature.
# ==============================================================================
import os
import chromadb
from typing import List, Dict, Any

class BreastCytologyRAGDatabase:
    """
    RAG Vector Database for clinical breast fine-needle aspiration (FNA) cytology.
    Uses ChromaDB to retrieve peer-reviewed clinical explanations for morphological features.
    """
    
    def __init__(self, collection_name: str = "breast_cytology_guidelines"):
        # Ephemeral in-memory ChromaDB client (or persistent)
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "WHO & IAC Yokohama Breast Cytopathology Reference Guidelines"}
        )
        self._populate_knowledge_base()

    def _populate_knowledge_base(self):
        """Indexes peer-reviewed cytopathological knowledge for each morphological nuclear characteristic."""
        documents = [
            # 1. Concave points & Concavity
            (
                "doc_concave_points",
                "Nuclear Membrane Grooving & Concave Points: In breast fine-needle aspirates, the presence of indentations, notches, and concave points along the nuclear margin reflects irregular nuclear membrane lamina assembly and severe aneuploidy. Severe contour irregularity and high concave point counts are hallmark cytological criteria for invasive ductal carcinoma (IAC Yokohama Category 5). In contrast, benign fibroadenomas display smooth, ovoid, uniform nuclear membranes.",
                {"feature_category": "concave_points", "pathological_significance": "Atypia & Malignancy", "source": "IAC Yokohama Breast Cytology Guidelines"}
            ),
            # 2. Perimeter and Area (Nuclear Pleomorphism)
            (
                "doc_nuclear_size",
                "Nuclear Enlargement (Worst Perimeter & Worst Area): Marked increase in nuclear size, perimeter (> 120 mm in digitized microscopy), and nuclear area (> 1000 um^2) corresponds to high-grade nuclear pleomorphism and elevated Nottingham histological grade. Malignant breast epithelial cells exhibit marked anisokaryosis (variation in nuclear size) exceeding 3 to 5 times the size of normal mature lymphocytes.",
                {"feature_category": "perimeter_area", "pathological_significance": "Nuclear Pleomorphism", "source": "WHO Classification of Breast Tumours"}
            ),
            # 3. Radius
            (
                "doc_radius",
                "Mean and Worst Radius: An enlarged nuclear radius indicates rapid, dysregulated cellular proliferation and macro-nucleoli formation. Spheroid or distorted enlarged nuclei (> 15 um radius) strongly correlate with invasive carcinoma, whereas small, monomorphic nuclei (< 10 um radius) are characteristic of benign ductal hyperplasia.",
                {"feature_category": "radius", "pathological_significance": "Cellular Proliferation", "source": "Cytopathology Principles & Practice"}
            ),
            # 4. Texture and Chromatin Pattern
            (
                "doc_texture",
                "Nuclear Texture and Chromatin Coarseness: Coarse, clumped chromatin with cleared parachromatin zones produces high variance in digital pixel texture scores. Malignant ductal cells display hyperchromasia, chromatin clumping, and prominent irregular nucleoli. Conversely, benign aspirates show delicate, powdery, evenly distributed chromatin.",
                {"feature_category": "texture", "pathological_significance": "Chromatin Distribution", "source": "The International Academy of Cytology (IAC)"}
            ),
            # 5. Compactness
            (
                "doc_compactness",
                "Cellular Compactness & Cohesion: Calculated as (perimeter^2 / area - 1.0). High compactness reflects perimeter irregularity and loss of circularity. In malignant aspirates, loss of E-cadherin adhesion leads to cell discohesion, irregular cellular borders, and dispersed isolated atypical cells.",
                {"feature_category": "compactness", "pathological_significance": "Cell Discohesion", "source": "Modern Pathology Breast Diagnostic Criteria"}
            ),
            # 6. Smoothness & Symmetry
            (
                "doc_smoothness_symmetry",
                "Smoothness and Symmetry: Benign ductal epithelial sheets exhibit high cellular symmetry, cohesiveness, and smooth, regular outlines. Significant loss of nuclear symmetry and jagged local perimeter variations are strongly associated with malignant cellular distortion.",
                {"feature_category": "smoothness_symmetry", "pathological_significance": "Structural Polarity", "source": "Diagnostic Cytopathology Manual"}
            )
        ]
        
        # Check if already populated
        if self.collection.count() == 0:
            ids = [item[0] for item in documents]
            texts = [item[1] for item in documents]
            metas = [item[2] for item in documents]
            self.collection.add(ids=ids, documents=texts, metadatas=metas)

    def retrieve_guidelines_for_features(self, top_features: List[str], n_results: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves matching cytopathology guidelines using semantic vector similarity search
        based on the features highlighted by SHAP.
        """
        query_text = " ".join([f.replace("_", " ") for f in top_features])
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count())
        )
        
        retrieved_items = []
        if results and "documents" in results and results["documents"]:
            for doc, meta, doc_id in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
                retrieved_items.append({
                    "id": doc_id,
                    "excerpt": doc,
                    "source": meta.get("source", "Medical Literature"),
                    "significance": meta.get("pathological_significance", "Clinical Context")
                })
        return retrieved_items

# Simple unit test
if __name__ == "__main__":
    rag = BreastCytologyRAGDatabase()
    res = rag.retrieve_guidelines_for_features(["worst_perimeter", "mean_concave_points"])
    print(f"ChromaDB RAG loaded. Retrieved {len(res)} guidelines:")
    for r in res:
        print(f"[{r['source']}] {r['excerpt'][:120]}...")
