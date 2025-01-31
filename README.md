# Article Abstract Classifier

This project is designed to classify article abstracts into three categories: Original, AI Generated and Reinterpreted. It uses various NLP techniques, embedding models and deep learning architectures for classification.

## Project Description

In this project, article abstracts obtained from the Journal Park website and covering five different topics were used. These abstracts were reinterpreted using five different LLM models (LLaMA, Gemma, Grok, ChatGPT and Gemini). In addition, AI-generated content was generated using the keywords of the articles. The dataset consists of 5,000 original abstracts, 25,000 paraphrased abstracts and 25,000 AI-generated abstracts, forming an unbalanced dataset.

After text preprocessing, these summaries were vectorized using five different embedding methods (Word2Vec, FastText, TF-IDF, T5 and BERT). The generated vectors were then classified using three deep learning algorithms (LSTM, CNN, Capsule Network) and two transformers (T5 and BERT). The model's performance was evaluated based on its ability to classify text according to three labels.

## Features

Text Preprocessing: tokenization, stopword removal, stemming and lemmatization using NLTK and Zemberek.

Word Embedding: Word2Vec, FastText, TF-IDF, BERT and T5 for text representation.

Machine Learning and Deep Learning Models:

LSTM, CNN, Capsule Network and BERT, T5 Transformer for classification.

TensorFlow and PyTorch applications.

Performance Metrics: Accuracy, Precision, Recall and F1-score.

