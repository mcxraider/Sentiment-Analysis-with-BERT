# Sentiment Analysis of Movie Reviews
## Introduction
This project explores the application of Machine Learning (ML) and Deep Learning (DL) techniques to perform sentiment analysis on movie reviews. The goal is to automatically classify the sentiment of written movie reviews as positive or negative, which can help in understanding audience preferences and improve film marketing strategies.

## Team Members
Jerry Yang
Felise Leow 
Ngu JiaHao 
Roydon Tay

## Table of Contents
Project Overview
Data Collection
Exploratory Data Analysis
Machine Learning Models
Deep Learning Models
Results
Usage
References

## Project Overview
We evaluated various ML and DL models using Python libraries like Scikit-Learn and PyTorch on a dataset of movie reviews. 
The techniques include:
Text preprocessing (TF-IDF and Count Vectorizer)
Random Forest, Multinomial Naive Bayes, and Logistic Regression
BERT and DistilBERT for advanced text representations
Data Collection
The dataset comprises user-generated movie reviews collected from various online platforms. Reviews were preprocessed to remove noise and formatted using techniques such as tokenization, lemmatization, and removal of stopwords.

## Exploratory Data Analysis
We performed n-gram analysis to identify common phrases in positive and negative reviews. EDA tools such as boxplots and countplots were used to understand distribution characteristics and sentiment bias in the data.

## Machine Learning Models
We implemented baseline models using:

Random Forest Classifier: An ensemble model to reduce overfitting and improve accuracy.
Multinomial Naive Bayes: A probabilistic model ideal for text data classification.
Logistic Regression: A fundamental model for binary classification tasks.
Model performance was enhanced using hyperparameter tuning with GridSearchCV and RandomizedSearchCV.

## Deep Learning Models
Our DL approach involved fine-tuning DistilBERT, a lighter version of BERT that retains most of the original model's predictive power but is less resource-intensive. This allows us to handle larger data sets more efficiently.

## Results
Our models achieved the following accuracy:

Best ML Model (Logistic Regression with TF-IDF): 91.0% Accuracy
Best DL Model (DistilBERT): 93.0% Accuracy
Detailed performance metrics including precision, recall, and F1-score are available in the results section of this repository.

## Usage
To replicate our findings or use the models:

Clone this repository.
Install the required packages from requirements.txt.
Run the Jupyter notebooks provided in the code/ directory to train the models on your data.

## References
Significant references include papers on BERT and DistilBERT, traditional ML techniques, and the latest advancements in sentiment analysis. For detailed citations, please refer to the references.md file.
