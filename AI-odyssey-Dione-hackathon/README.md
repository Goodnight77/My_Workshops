# AI-odyssey-Dione-hackathon

This project demonstrates an **Agentic RAG (Retrieval-Augmented Generation)** system. It showcases how to build intelligent agents capable of retrieving information and performing tasks using vector search and large language models.

## Tools & Technologies
The notebook utilizes the following libraries and tools:
- **smolagents**: Framework for building code-writing agents.
- **Qdrant**: Vector database for efficient similarity search and retrieval.
- **Chonkie**: A library for text chunking and data preparation (`TextChef`, `RecursiveChunker`).
- **Hugging Face Transformers**: Uses models like `HuggingFaceTB/SmolLM3-3B`.
- **LangChain**: Integrations for Qdrant and OpenAI (`langchain-qdrant`, `langchain_openai`).
- **Rich**: For beautiful terminal formatting and markdown rendering.
- **PyPDF**: For processing PDF documents (if you need to use PDFs instead of md files)

## Contents
- `Smollgents_with_Qdrant.ipynb` : main notebook/demo using Qdrant.
- `resources.md` : additional resources and links.

## Usage
1. Open the notebook with Jupyter or Google colab.
2. (Optional) Create a virtual environment and install dependencies.
	 - Windows (PowerShell):
		 ```powershell
		 python -m venv .venv
		 .\.venv\Scripts\Activate.ps1
		 pip install -r requirements.txt
		 ```
3. Run the notebooks interactively.

## Notes
- For questions please feel free to reach out on linkedin [Mohamed Arbi Nsibi](https://www.linkedin.com/in/mohammed-arbi-nsibi-584a43241/)