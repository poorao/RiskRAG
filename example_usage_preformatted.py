from rag_framework import RAGFramework

if __name__ == "__main__":
    # Initialize RAG framework
    rag = RAGFramework(
        retriever_model="BAAI/bge-large-en-v1.5",
        api_key="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",
        max_len=512,
        top_k=5
    )

    # Input model description
    model_description = ["This is a model for text classification in social media contexts.", 
                         "Orca 2 is a finetuned version of LLAMA-2. Orca 2‚Äôs training data is a synthetic dataset that was created to enhance the small model‚Äôs reasoning abilities. "
                         "All synthetic training data was moderated using the Microsoft Azure content filters. More details about the model can be found in the [Orca 2 paper](https://arxiv.org/pdf/2311.11045.pdf)"]

    # File paths
    risk_cards_file = "/Users/admin/Code/git/AI-model-card-analysis-HuggingFace/from_server/new_data/cards_with_formatted_risks.csv"
    embeddings_file = "embeddings/corpus_embeddings"
    # extracted_risks_file = "data/extracted_risks.json"

    # Process and retrieve risks
    risks = rag.process_preformatted(model_description, risk_cards_file, embeddings_file)
    print(risks)