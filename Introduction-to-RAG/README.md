### 2. [Introduction to Retrieval-Augmented Generation (RAG)](02-into-to-RAG/)


## Overview

This workshop provides a gentle introduction to Retrieval-Augmented Generation (RAG) systems using an open-source model, specifically Llama 3.1. Participants will learn how to create a simple RAG application that answers questions from a PDF document.

## Objectives

By the end of this workshop, participants will be able to:

- Understand the concept of Retrieval-Augmented Generation.
- Set up a Python environment with the necessary libraries.
- Load and preprocess a PDF document.
- Split the document into manageable chunks.
- Store these chunks in a vector store.
- Retrieve information based on user queries.

## Requirements

Participants should have:

- Basic knowledge of Python programming.
- Anaconda or Miniconda installed.
- Familiarity with Jupyter Notebooks.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Goodnight77/GenerativeAI_bootcamp.git
   cd 02-into-to-RAG
2. **Create a Conda Environment**
    ```bash
    conda create -n rag python=3.10
    conda activate rag
3. **Install Required Libraries**
    ```bash
    pip install torch transformers langchain-community langchain-huggingface notebook==7.1.2 pandas faiss-cpu scikit-learn
4. **Alternatively, you can install all the requirements using the requirements.txt file:**
    ```bash
      pip install -r requirements.txt
5. **Install Ollama for Local Models (Optional)**
If you wish to use local models with Ollama, follow the steps below to install and configure Ollama.

Download Ollama from the official website: ollama.com.

Install Ollama following the instructions for your operating system.

6. **Run the Jupyter Notebook**
    ```bash
    jupyter notebook
7. **Open the notebook RAG.ipynb**

## Homework for next session
you can find in the HOMEWORK folder a homework notebook for the next workshop 

### other resources
In the slides folder a presentation where you can also find all some RAG ressources 


