# Sentiment Analysis of Movie Reviews
## Introduction
This project explores the application of Machine Learning (ML) and Deep Learning (DL) techniques to perform sentiment analysis on movie reviews. The goal is to automatically classify the sentiment of written movie reviews as positive or negative, which can help in understanding audience preferences and improve film marketing strategies.

## Team Members
- [Jerry Yang](https://github.com/mcxraider)
- [Ngu JiaHao](https://github.com/yjiahao)
- [Roydon Tay](https://github.com/RoydonTay)
- Felise Leow 

## Project Overview
We evaluated various ML and DL models using Python libraries like Scikit-Learn and PyTorch on a dataset of movie reviews. 
The techniques include:
- Text preprocessing (TF-IDF and Count Vectorizer)
- Random Forest, Multinomial Naive Bayes, and Logistic Regression
- BERT and DistilBERT for advanced text representations

The dataset comprises user-generated movie reviews collected from various online platforms. Reviews were preprocessed to remove noise and formatted using techniques such as tokenization, lemmatization, and removal of stopwords. Experiements were conducted for 8-class, 3-class and 2-class classification for ML methods, and 8-class and 2-class for DL, with 2-class yielding highest accuracies in both cases.

## Exploratory Data Analysis
We performed n-gram analysis to identify common phrases in positive and negative reviews. EDA tools such as boxplots and countplots were used to understand distribution characteristics and sentiment bias in the data.

## Machine Learning Models
We implemented baseline models using:

- Random Forest Classifier: An ensemble model to reduce overfitting and improve accuracy.
- Multinomial Naive Bayes: A probabilistic model ideal for text data classification.
- Logistic Regression: A fundamental model for binary classification tasks.
Model performance was enhanced using hyperparameter tuning with GridSearchCV and RandomizedSearchCV.

## Deep Learning Models
Our DL approach involved fine-tuning DistilBERT, a lighter version of BERT that retains most of the original model's predictive power but is less resource-intensive. This allows us to handle larger data sets more efficiently.

## Results
Our models achieved the following accuracy:

| Model  | Accuracy|
| -------- | ------- |
| Best ML Model (Logistic Regression with TF-IDF) | 91.0%  |
| Best DL Model (DistilBERT with finetuning) | 93.0%    |

Detailed performance metrics including precision, recall, and F1-score are available in the results section of this repository.

## Usage
To replicate our findings or use the models:

Clone this repository.
Install the required packages from requirements.txt.
Run the Jupyter notebooks provided in the code/ directory to train the models on your data.
Alternatively, you may run the notebooks on Google Colaboratory from the links in the respective notebooks.
