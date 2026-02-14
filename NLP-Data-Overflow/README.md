# Advanced NLP: From Sequential Bottlenecks to Parallel Architectures

[![Presenter: Mohamed Arbi Nsibi](https://img.shields.io/badge/Presenter-Mohamed%20Arbi%20Nsibi-blue)](https://www.linkedin.com/in/mohammed-arbi-nsibi-584a43241/)
[![Bootcamp: Data Overflow](https://img.shields.io/badge/Bootcamp-Data%20Overflow-orange)]()


## Overview
This repository contains the complete materials for the **Advanced NLP Workshop** presented at the **Data Overflow Bootcamp**. 

Moving beyond the basics of "how to use a model," this project dives deep into the **engineering mechanics, architectural trade-offs, and history** required for modern production-grade NLP. It covers the evolutionary journey from bottlenecked recurrent systems to the parallelized Transformer architectures used by GPT and Gemini.

---

## Key Technical Pillars

### 1. High-Signal Text Processing & Cleaning
* **The Preprocessing Pipeline:** Mastering the flow from raw data acquisition and cleaning to feature engineering.
* **Linguistic Analysis:** POS tagging and Named Entity Recognition (NER) to extract structured meaning from raw strings.
* **Subword Tokenization:** Using algorithms like **BPE and WordPiece** to handle neologisms and eliminate the destructive `[UNK]` token.
* **Real-World Robustness:** Why Byte-Level BPE is mandatory for handling noisy user-generated content like slang and emojis.

### 2. Neural Representations & Vector Geometry
* **From Sparse to Dense:** Transitioning from traditional methods (**One-Hot Encoding, TF-IDF**) to dense **Word Embeddings**.
* **Semantic Arithmetic:** Visualizing high-dimensional relationships (e.g., **King - Man + Woman = Queen**) and clustering similar concepts in 2D space.
* **Contextual Evolution:** Comparing static models like **Word2Vec and GloVe** against contextual powerhouses like **BERT**.
* **Similarity Calibration:** Using **Cosine and Jaccard Similarity** to mathematically quantify semantic overlap.

### 3. The Sequential Crisis
* **RNN/LSTM Mechanics:** Analyzing how hidden states act as "memory" for sequential data.
* **The Vanishing Gradient:** Why the exponential decay of gradients prevents RNNs from learning long-range dependencies.
![alt text](/assets/vanishing_gradient.png)

* **The Information Bottleneck:** The failure of forcing entire sequences into one fixed-length vector, leading to data overflow in long sentences.


### 4. Transformer Architecture (The Real Mechanics)
* **The Attention Engine:** Deep dive into **Queries (Q), Keys (K), and Values (V)** and the scaled dot-product attention formula.
![alt text](/assets/attention.png)

* **Geometric Stability:** The critical role of the **$\sqrt{d_k}$ scaling factor** in preventing gradient saturation.
* **Positional Vision:** Injecting spatial order into permutation-invariant models using periodic Sin/Cos functions.
* **Softmax Temperature:** Visualizing how temperature (T) controls the "focus" of attention scores.

### 5. Task Specialization: BERT vs. GPT
* **Encoder vs. Decoder:** Understanding the **Expert Reader (BERT)** bidirectional context vs. the **Expert Writer (GPT)** autoregressive generation.
* **Use Case Alignment:** Applying masked self-attention for generative tasks vs. full self-attention for classification and understanding.

## Hands-on Notebooks
The curated notebooks that illustrate the end-to-end pipeline are available here:
* [NLP pipeline walkthrough](https://github.com/Goodnight77/My_Workshops/blob/main/Into-to-NLP-GODS-4.0/NLP_pipeline.ipynb)
* [NLP preprocessing labs](https://github.com/Goodnight77/My_Workshops/blob/main/Into-to-NLP-GODS-4.0/NLP_preprocessing.ipynb)


## Resources
* [RNN lecture by Huang Xiao (The Hong Kong Polytechnic University)](https://www4.comp.polyu.edu.hk/~xiaohuang/docs/COMP4434_slides/Lecture10_Recurrent_Neural_Networks.pdf)
* [OpenAI tokenizer explorer](https://platform.openai.com/tokenizer)
* [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
* [Generating Fake News with OpenAI’s Language Models](https://medium.com/data-science/creating-fake-news-with-openais-language-models-368e01a698a3)
* [What’s before GPT-4? A deep dive into ChatGPT](https://medium.com/digital-sense-ai/whats-before-gpt-4-a-deep-dive-into-chatgpt-dfce9db49956)
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

--- 
### Special Thanks
A special thank you to the organizers and participants of the **Data Overflow Bootcamp**. Your commitment to high-level technical excellence made this workshop possible.
