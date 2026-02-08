# Intro to Retrieval-Augmented Generation (RAG) Data Overflow

## Overview
This project provides an introduction to Retrieval-Augmented Generation (RAG) through a Jupyter Notebook and an accompanying presentation. It demonstrates a basic RAG data flow, illustrating how retrieval-based augmentation enhances generative models.

## Features
- **Jupyter Notebook (`basic_rag_dataflow.ipynb`)**: A hands-on demonstration of RAG data flow with step-by-step explanations.
- **Presentation Slides (`slides/Introduction to RAG.pdf`)**: A conceptual overview of RAG, explaining key principles and benefits.
- **Google Colab Support**: Run the notebook directly in Google Colab without local installation. [Open Google Colab](https://colab.research.google.com/drive/133FQCGqZKXWrFe4RYDJIjW-ST9P78AH1?usp=sharing) and then upload this notebook.

## Installation
This project requires Python and some essential libraries. you can install dependencies manually within the notebook itself.

### **Step-by-step setup:**
1. **Install Python** (if not already installed) from [python.org](https://www.python.org/).
2. **Install Jupyter Notebook** by running:
   ```sh
   pip install jupyter
   ```
3. **Install dependencies** (inspect `basic_rag_dataflow.ipynb` for required packages). Common ones include:
   ```sh
   pip install pandas langchain langchain-groq
   ```

## Usage
### **Local Execution**
1. **Navigate to the project directory**:
   ```sh
   cd Intro-RAG-Data-Overflow
   ```
2. **Start Jupyter Notebook**:
   ```sh
   jupyter notebook
   ```
3. **Open and run the notebook**: In the Jupyter interface, open `basic_rag_dataflow.ipynb` and execute the cells sequentially.
4. **View the slides**: Open `slides/Introduction to RAG.pdf` for the presentation.

### **Run on Google Colab**
If you prefer not to set up a local environment, open the notebook in Google Colab:
[Run Google Colab](https://colab.research.google.com/)

## Resources
The following resources are discussed in the presentation:
- **Your RAG powered by Google Search Technology**: [link](https://cloud.google.com/blog/products/ai-machine-learning/rags-powered-by-google-search-technology-part-1?hl=en&linkId=9446045)
- **Let’s talk about LlamaIndex and LangChain** [link](https://superwise.ai/blog/lets-talk-about-llamaindex-and-langchain/#:~:text=On%20the%20one%20hand%2C%20LlamaIndex,driven%20pipelines%20for%20streamlined%20operations.)
- **Retrieval-Augmented Generation (RAG) framework in Generative AI**[link](https://www.linkedin.com/pulse/retrieval-augmented-generation-rag-framework-generative-raja-zjasc/)
- **Retrieval-Augmented Generation (RAG): From Theory to LangChain Implementation**[link](https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2/)
- **Advanced Retrieval-Augmented Generation: From Theory to LlamaIndex Implementation**[link](https://towardsdatascience.com/advanced-retrieval-augmented-generation-from-theory-to-llamaindex-implementation-4de1464a9930/#c1e2)
- **Retrieval-augmented generation for large language models: A survey** [link](https://arxiv.org/pdf/2312.10997)

## Additional Notes
- You need to use GROQ API, you can get an API key from [GROQ's official website](https://console.groq.com/keys) and configure it in your environment.
- Modify the notebook as needed to experiment with different retrieval and generation approaches.

## Acknowledgment

Special thanks to Data Overflow for the opportunity to conduct this workshop at INSAT.

## License
This project is open-source. Feel free to modify and extend it!

