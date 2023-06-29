# Sentiment-Analysis-on-twitter-dataset_sobika

# Sentiment Analysis on Twitter Dataset

This project focuses on performing sentiment analysis on the Twitter dataset using various machine learning models such as SVM (Support Vector Machine) and Logistic Regression. Additionally, the performance of the models is compared with the ROBERTA model for sentiment analysis.

## Dataset

The Twitter Sentiment Dataset is used for this project. It contains a collection of tweets along with sentiment scores. The dataset is preprocessed to remove stopwords, punctuation, URLs, and repeating characters. It is also subjected to lemmatization and stemming for better analysis.

## Models

1. SVM (Support Vector Machine): The SVM model is trained on the preprocessed Twitter dataset using the TF-IDF vectorizer. The trained model is then evaluated on the test set to measure its performance.

2. Logistic Regression: Similar to the SVM model, the Logistic Regression model is trained on the preprocessed Twitter dataset using TF-IDF vectorization. The model's performance is evaluated on the test set.

3. ROBERTA Model: In addition to the traditional machine learning models, the ROBERTA model is employed to test and match the performance of the SVM and Logistic Regression models. The ROBERTA model uses contextual embeddings to analyze the sentiment of input sentences.

## Usage

1. Preprocessing: The dataset is preprocessed using several functions such as removing stopwords, punctuation, URLs, and repeating characters. Lemmatization and stemming techniques are also applied for text normalization.

2. SVM and Logistic Regression Models: The preprocessed dataset is split into training and testing sets. The SVM and Logistic Regression models are trained on the training set using the TF-IDF vectorization. The trained models are then evaluated on the test set to measure their performance.

3. ROBERTA Model: The ROBERTA model is used to test and match the performance of the SVM and Logistic Regression models. Example sentences are preprocessed and passed through the ROBERTA model for sentiment analysis.

## Results

The performance of the SVM and Logistic Regression models is evaluated using metrics such as accuracy, precision, recall, and F1-score. The ROBERTA model's predictions are compared with the results obtained from the traditional machine learning models.

## Conclusion

This project demonstrates the application of machine learning models, specifically SVM and Logistic Regression, for sentiment analysis on the Twitter dataset. The performance of these models is compared with the ROBERTA model, showcasing the effectiveness of different approaches in sentiment analysis tasks.

Please refer to the project code and documentation for more details on the implementation and usage.


