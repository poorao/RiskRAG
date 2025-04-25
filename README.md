# RiskRAG

RiskRAG is a repository associated with the CHI 2025 paper **"RiskRAG: A Data-Driven Solution for Improved AI Model Risk Reporting"**. This is a data-drive solution that uses a retrieval-augmented generation framework for AI model risk reporting. RiskRAG is designed to support developers and practitioners in identifying risks of AI models in understandable, clear language along with associated mitigations strategies to produce an actionable risk report. 

Project details available at: [https://social-dynamics.net/ai-risks/card/](https://social-dynamics.net/ai-risks/card/)

[![Paper](https://img.shields.io/badge/paper-ACM%20DOI%3A10.1145%2F3706598.3713979-blue)](https://dl.acm.org/doi/10.1145/3706598.3713979)
[![arXiv](https://img.shields.io/badge/arXiv-2504.08952-<COLOR>.svg)](https://arxiv.org/pdf/2504.08952)

## Overview

The framework consists of two main components:

1. **Retriever**: Retrieves relevant risk sections from a pre-existing dataset of model cards and descriptions from AI incidents.
2. **Generator**: Generates risks in the desired standardized format of verb + object + [explanation], starting with an action verb.

The repository supports two modes of operation:
- **Retriever-only mode**: We provide a dataset of model cards with risks already pre-formatted in the desired standard at `data/cards_with_formatted_risks.csv`. You can directly retrieve risks using the retriever without needing to run language models again, minimising environmental impact and computational overhead. 
- **Retriever + Generator mode**: For new datasets, the framework generates risks in standard format using the generator and creates embeddings for efficient retrieval.

## Features

- **Risk Retrieval and Generations**: Retrieve top-k risks and generate in a structured format from a dataset of model cards based on model descriptions.
- **Embeddings Support**: Automatically compute embeddings for new datasets to enable efficient retrieval.


## Installation

1. Clone the repo:
   ```bash
   git clone git@github.com:poorao/RiskRAG.git
   cd RiskRAG
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure your Hugging Face token is available as an environment variable:  
   ```bash
   export HUGGINGFACE_TOKEN="your_token_here"
   ```  

## Usage

### 1. Retriever-Only Mode

If you would like to use preformatted dataset with risks (`cards_with_formatted_risks.csv`), you can use the retriever to retrieve risks and mitigations based on model descriptions. An example usage is provided in [`example_usage_preformatted.py`](example_usage_preformatted.py):

```python
from rag_framework import RAGFramework

if __name__ == "__main__":
    # Initialize RAG framework
    rag = RAGFramework(
        retriever_model="BAAI/bge-large-en-v1.5",
        api_key="your_openai_api_key",
        max_len=512,
        top_k=5
    )

    # Input model description
    model_description = ["This is a model for text classification in social media contexts."]

    # File paths
    risk_cards_file = "path/to/cards_with_formatted_risks.csv"
    embeddings_file = "embeddings/corpus_embeddings"

    # Retrieve risks
    risks = rag.process_preformatted(model_description, risk_cards_file, embeddings_file)
    print(risks)
```

### 2. Retriever + Generator Mode
For new datasets, the framework generates risks and computes embeddings. An example usage is provided in example_usage.py:

```python
from rag_framework import RAGFramework

if __name__ == "__main__":
    # Initialize RAG framework
    rag = RAGFramework(
        retriever_model="BAAI/bge-large-en-v1.5",
        api_key="your_openai_api_key",
        max_len=512,
        top_k=5
    )

    # Input model description
    model_description = ["This is a model for text classification in social media contexts."]

    # File paths
    risk_cards_file = "path/to/new_dataset.csv"
    embeddings_file = "embeddings/new_embeddings"

    # Generate and retrieve risks
    risks = rag.process(model_description, risk_cards_file, embeddings_file)
    print(risks)
```

### Embeddings
Embeddings are computed using the BAAI/bge-large-en-v1.5 model and stored in the embeddings/ directory. Change the retriever_model to generate embeddings using any other model.

## Coming Soon
- Support to include risks from AI incidents dataset
- Code to replicate results from the *Baseline Evaluation* in the paper
- Risk categories from DeepMind Risk taxonomy

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{rao2025riskrag,
  author    = {Rao, Pooja S.~B. and {\v{S}}{\'c}epanovi{\'c}, Sanja and Zhou, Ke and Bogucka, Edyta and Quercia, Daniele},
  title     = {{RiskRAG}: {A} Data-Driven Solution for Improved {AI} Model Risk Reporting},
  booktitle = {Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI ’25)},
  pages     = {1--26},
  year      = {2025},
  month     = apr,
  address   = {Yokohama, Japan},
  publisher = {ACM},
  doi       = {10.1145/3706598.3713979}
}
```

