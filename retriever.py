import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process, utils
from get_embeddings import compute_embeddings

class RiskRetriever:
    def __init__(self, model, max_len=512, top_k=5):
        self.model = model
        self.max_len = max_len
        self.top_k = top_k


    def retrieve_topk_matches(self, cosine_sim, k=10):
        """Retrieve top-k matches based on cosine similarity."""
        return np.argsort(cosine_sim, axis=1)[:, -k:][:, ::-1]

    def retrieve_risks(self, model_description, risk_cards, embeddings_file):
        """Retrieve risks for a given model description."""
        # Compute embeddings
        embeddings_corpus, embeddings_queries = compute_embeddings(
            corpus=risk_cards["model_description"].tolist(),
            queries=model_description,
            model=self.model,
            max_len=self.max_len,
            batch_size=8,
            embeddings_file=embeddings_file
        )

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(embeddings_queries, embeddings_corpus)
        topk_indices = self.retrieve_topk_matches(similarity_matrix, k=self.top_k)

        # Retrieve risks
        retrieved_risks = []
        for idx in topk_indices[0]:
            retrieved_risks.append({
                "description": risk_cards.iloc[idx]["model_description"],
                "risk_section": risk_cards.iloc[idx]["risks_limitations_bias"],
                "risks": risk_cards.iloc[idx]["risks"],
                "mitigations": risk_cards.iloc[idx]["mitigations"]
            })
        return retrieved_risks