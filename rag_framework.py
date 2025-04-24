import pandas as pd
from retriever import RiskRetriever
from generate import RiskGenerator

class RAGFramework:
    def __init__(self, retriever_model, api_key, max_len=512, top_k=5):
        self.retriever = RiskRetriever(model=retriever_model, max_len=max_len, top_k=top_k)
        self.generator = RiskGenerator(api_key=api_key)

    def process(self, model_description, risk_cards_file, embeddings_file=None):
        """Main process to retrieve and generate risks."""
        # Load data
        risk_cards = pd.read_csv(risk_cards_file)

        # Retrieve risks
        retrieved_risks = self.retriever.retrieve_risks(model_description, risk_cards, embeddings_file)

        # Generate risks
        generated_risks = []
        for i, risk in enumerate(retrieved_risks):
            generated_risks.append(self.generator.generate_risks(i, risk["risk_section"]))

        return generated_risks
    
    def process_preformatted(self, model_description, risk_cards_file, embeddings_file=None):
        """Main process to retrieve and generate risks."""
        # Load data
        risk_cards = pd.read_csv(risk_cards_file)

        # Retrieve risks
        retrieved_risks = self.retriever.retrieve_risks(model_description, risk_cards, embeddings_file)

        return retrieved_risks