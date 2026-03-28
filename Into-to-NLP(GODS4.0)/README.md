# Introduction to NLP (GODS) 4.0

## Project Overview

This project, "Introduction to NLP (GODS) 4.0," provides a comprehensive introduction to Natural Language Processing (NLP). It includes Jupyter Notebooks for building an NLP pipeline and preprocessing text data, along with supporting data and presentation slides.

## Features

* **NLP Pipeline:** A Jupyter Notebook (`NLP_pipeline.ipynb`) demonstrating the construction of a complete NLP pipeline (NER,BOW,TF-IDF,Word2vec,..).
* **NLP Preprocessing:** A Jupyter Notebook (`NLP_preprocessing.ipynb`) detailing various text preprocessing techniques (lowercasing,stop-word removal,tokenization,data cleaning,.. ).
* **Data Directory:**  A dedicated directory (`data/`) containing the datasets used in the project.
* **Presentation Slides:** A PDF file (`slides/Into-to-NLP-(GODS).pdf`) summarizing the key concepts and findings.


## Installation

This project utilizes Python and Jupyter Notebooks.  The specific dependencies are not explicitly listed (e.g., no `requirements.txt` file is present).  To ensure a consistent environment, it is recommended to create a virtual environment.

**Using conda:**

1. Create a conda environment:  `conda create -n gods_nlp python=3.10` (adjust Python version as needed)
2. Activate the environment: `conda activate gods_nlp`
3. Install Jupyter Notebook: `conda install -c conda-forge notebook`
4. Navigate to the project directory and launch Jupyter Notebook: `jupyter notebook`


**Using virtualenv (alternative):**

1. Create a virtual environment: `python3 -m venv gods_nlp_env` (replace `gods_nlp_env` with your preferred environment name)
2. Activate the environment:
   * **Linux/macOS:** `source gods_nlp_env/bin/activate`
   * **Windows:** `gods_nlp_env\Scripts\activate`
3. Install Jupyter Notebook: `pip install notebook`
4. Navigate to the project directory and launch Jupyter Notebook: `jupyter notebook`

**Note:** You will likely need to install additional Python packages based on the specific NLP libraries used within the notebooks.  These will need to be identified from within `NLP_pipeline.ipynb` and `NLP_preprocessing.ipynb`.


## Usage

1. Activate your chosen virtual environment (as described in the Installation section).
2. Launch Jupyter Notebook from the project's root directory.
3. Open and run `NLP_pipeline.ipynb` and `NLP_preprocessing.ipynb` to explore the NLP pipeline and preprocessing techniques.
4. Refer to the `slides/Into-to-NLP-(GODS).pdf` for a summary of the project.


## Acknowledgments

I would like to extend my gratitude to **IEEE ENSI** for recording the workshop. You can find the recording of the workshop here: [Workshop Recording](https://ensi.ieee.tn/GODS/Bootcamp).

Additionally, I have written articles on Medium that provide further insights into NLP concepts:
- **Understanding TF-IDF: Term Frequency-Inverse Document Frequency**: [Read on Medium](https://medium.com/@mohammedarbinsibi/understanding-tf-idf-term-frequency-inverse-document-frequency-9d9299af1d87).
- **A Deep Dive into Natural Language Processing (NLP)**: [Read on Medium](https://medium.com/@mohammedarbinsibi/a-deep-dive-into-natural-language-processing-nlp-da0c0e30d71c).

