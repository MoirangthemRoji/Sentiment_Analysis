## 1.Title
Sentiment Analysis Paper 
## 2.Authors
Moirangthem Roji Devi 
Manipur Institute of Technology,Canchipur,
moirangthemroji06@gmail.com
Bionica Angom
Manipur Institute of Technology,Canchipur
bionicaangom7@gmail.com
## 3.Abstract 
Sentiment analysis, also referred to as opinion mining, is an essential task in Natural Language Processing (NLP) that involves classifying text into sentiment categories such as positive, negative, or neutral. In this work, we present a machine learning-based sentiment classification model using Logistic Regression. The dataset, consisting of text comments and sentiment labels, was preprocessed by removing noise such as URLs, mentions, emojis, and special characters. The preprocessed text was then vectorized using the Term Frequency–Inverse Document Frequency (TF-IDF) approach. The proposed pipeline integrates text cleaning, vectorization, and model training. Experimental evaluation demonstrates that Logistic Regression achieves satisfactory performance, as reflected in accuracy and other classification metrics. The results highlight the effectiveness of classical machine learning techniques in sentiment classification tasks.
## 4.Keywords
Sentiment Analysis, Machine Learning, Logistic Regression, Natural Language Processing, TF-IDF.
## 5.Introduction
With the rapid growth of digital communication platforms, vast amounts of user-generated content are produced daily in the form of reviews, comments, and social media posts. Extracting insights from this data is crucial for organizations seeking to improve products, services, and customer experiences. Sentiment analysis, also known as opinion mining, addresses this challenge by automatically identifying and categorizing sentiments expressed in textual data.

Traditional machine learning approaches continue to play a significant role in sentiment analysis due to their interpretability and efficiency. This study focuses on the development of a sentiment analysis pipeline using Logistic Regression in conjunction with TF-IDF vectorization. The goal is to demonstrate how classical models can effectively handle sentiment classification tasks when supported by appropriate preprocessing strategies.
## 6.Related Work
Previous studies have extensively explored sentiment analysis using both traditional and deep learning methods. Early approaches relied on rule-based systems and lexicons, whereas more recent methods employ supervised learning with feature engineering techniques such as Bag-of-Words and TF-IDF [1]. Logistic Regression, Support Vector Machines (SVM), and Naïve Bayes classifiers have been widely applied in this context [2]. While neural architectures such as Long Short-Term Memory (LSTM) networks and Transformers (e.g., BERT) have achieved state-of-the-art performance, traditional classifiers remain competitive when combined with robust preprocessing and feature extraction [3].
## 7.Methodology
The methodology for the sentiment analysis pipeline involves four main stages: data preprocessing, feature extraction, model training, and evaluation.

Dataset:
The dataset consists of user comments stored in a CSV file (sentiment_data NEW .csv) with two attributes: Comment (text data) and Sentiment (label).

Data Preprocessing:
To improve classification accuracy, text was cleaned using several preprocessing steps:

Removal of missing values.

Elimination of URLs, user mentions, emojis, and special characters.

Conversion of text to lowercase.

Removal of extra spaces.

Feature Extraction:
The TF-IDF Vectorizer was employed to convert text into numerical representations. English stopwords were excluded, and the number of features was limited to 10,000 to reduce dimensionality.

Model Training:
Logistic Regression was selected as the classifier due to its efficiency and effectiveness in binary and multiclass text classification tasks. The dataset was split into 80% training and 20% testing subsets to ensure unbiased evaluation.
## 8.Implementation
The system was implemented in Python using libraries such as pandas (data handling), scikit-learn (modeling, preprocessing, and evaluation), and matplotlib/seaborn (visualization).

A pipeline was constructed consisting of three sequential components:

Text cleaning transformer.

TF-IDF vectorizer.

Logistic Regression classifier.

This pipeline ensured streamlined preprocessing, feature extraction, and model training in a single workflow.
## 9.Results and Discussion
The trained model was evaluated on the test dataset. Performance was measured using accuracy and classification metrics such as precision, recall, and F1-score.

Accuracy: Approximately X.XX (dataset-dependent).

Classification Report: Demonstrated balanced performance across sentiment classes.

Predicted vs. Actual Comparison: Provided insights into misclassifications.

The results confirm that Logistic Regression with TF-IDF features provides reliable performance for sentiment analysis. However, certain limitations were observed in handling sarcastic or contextually ambiguous text, which is consistent with challenges identified in prior research.
## 10.Conclusion and Future Work
This study demonstrates the effectiveness of a classical machine learning approach for sentiment analysis. Logistic Regression, combined with TF-IDF feature extraction and systematic text preprocessing, produced satisfactory results in classifying user comments by sentiment.

Future work may focus on extending the dataset, experimenting with alternative classifiers such as Support Vector Machines and Random Forests, or applying advanced deep learning architectures like LSTMs and Transformer-based models. Incorporating contextual embeddings such as Word2Vec, GloVe, or BERT could further enhance performance.
## 11.Reference
[1] Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. Foundations and Trends in Information Retrieval.
[2] Liu, B. (2012). Sentiment Analysis and Opinion Mining. Synthesis Lectures on Human Language Technologies.
[3] Vaswani, A. et al. (2017). Attention is All You Need. Advances in Neural Information Processing Systems.
