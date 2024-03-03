# Sentiment Analysis of Movie Reviews
---

## Overview

This project applies natural language processing (NLP) techniques to analyze sentiment in movie reviews. The goal is to classify reviews into sentiment categories, providing insights into overall viewer reception. The project is structured into several Jupyter notebooks, each with a specific role in the data science workflow.

## Notebooks

### `Data_preprocessing.ipynb`

- **Purpose**: This notebook is dedicated to loading and preprocessing the dataset. It includes initial data exploration and cleaning steps such as handling missing values, removing unnecessary tags, and standardizing text.
- **Processes**:
  - Data importation from various sources
  - Preliminary data analysis for structure and content
  - Data cleaning and preprocessing

### `EDA.ipynb`

- **Purpose**: This notebook contains exploratory data analysis (EDA) to uncover patterns, spot anomalies, and test hypotheses. It includes visualizations and statistical analysis to understand the data better.
- **Processes**:
  - N-gram analysis to identify common phrases
  - Sentiment distribution analysis
  - Boxplot visualizations for review lengths and other relevant metrics

### `ML_experimentation.ipynb`

- **Purpose**: This notebook is used for machine learning experimentation. It includes model selection, feature engineering, hyperparameter tuning, and evaluation using various metrics.
- **Processes**:
  - Splitting the dataset into training and testing sets
  - Vectorization of textual data and feature extraction
  - Training different ML models like Naive Bayes, SVM, and Random Forest
  - Model evaluation and comparison

### `DL_experimentation.ipynb`

- **Purpose**: This notebook focuses on deep learning experiments using neural networks. It explores the use of pre-trained models like BERT for sentiment analysis.
- **Processes**:
  - Tokenization and encoding of text data for BERT
  - Building and training custom neural network architectures
  - Fine-tuning pre-trained models and evaluating their performance

## Dataset

The dataset consists of movie reviews, each labeled with its corresponding sentiment. It has been preprocessed to remove stop words while retaining those important for sentiment analysis.

## Libraries Used

- NLP: NLTK
- Machine Learning: scikit-learn
- Deep Learning: PyTorch, Transformers
- Data Manipulation: pandas, NumPy, regex
- Visualization: matplotlib, seaborn

## Installation

Provide steps for installing required libraries and setting up the environment, for example:

```bash
pip install -r requirements.txt
```

## Contributors:


## Acknowledgements:
Research Papers:

