# Sentiment-Analysis-on-twitter-dataset_sobika

# Sentiment Analysis on Twitter Dataset

This project focuses on performing sentiment analysis on the Twitter dataset using various machine learning models such as SVM (Support Vector Machine) and Logistic Regression. Additionally, the performance of the models is compared with the ROBERTA model for sentiment analysis.

## Dataset

The Twitter Sentiment Dataset is used for this project. It contains a collection of tweets along with sentiment scores. The dataset is preprocessed to remove stopwords, punctuation, URLs, and repeating characters. It is also subjected to lemmatization and stemming for better analysis.


## Models

1. SVM (Support Vector Machine): The SVM model is trained on the preprocessed Twitter dataset using the TF-IDF vectorizer. The trained model is then evaluated on the test set to measure its performance.

2. Logistic Regression: Similar to the SVM model, the Logistic Regression model is trained on the preprocessed Twitter dataset using TF-IDF vectorization. The model's performance is evaluated on the test set.

3. Naive Bayes Model
   
4. ROBERTA Model: In addition to the traditional machine learning models, the ROBERTA model is employed to test and match the performance of the SVM and Logistic Regression models. The ROBERTA model uses contextual embeddings to analyze the sentiment of input sentences.

# Sentiment Analysis

This project focuses on sentiment analysis using various machine learning models. It includes training and testing notebooks to preprocess the data, train the models, and evaluate their performance.

## Files

1. Training Notebook: `train.ipynb`
   - This notebook contains the following steps:
     - Preprocessing the training data
     - Transforming the data using TF-IDF vectorization
     - Training SVM, Logistic Regression, Naive Bayes, and Roberta models
     - Storing the trained models using `joblib.dump`

2. Testing Notebook: `test.ipynb`
   - This notebook contains the following steps:
     - Preprocessing the testing data
     - Transforming the data using TF-IDF vectorization
     - Importing the trained SVM, Logistic Regression, and Naive Bayes models
     - Printing the predicted target labels and classification report for evaluation

## Usage

1. Open and run the `train` notebook to train the sentiment analysis models. Make sure to provide the necessary training data and adjust the preprocessing and model training steps as needed.

2. Once the models are trained, open and run the `test` notebook to evaluate the models on the testing data. Update the notebook with the appropriate testing data and adjust the preprocessing and model import steps accordingly.

Note: The trained models are stored using `joblib.dump` in the training notebook and imported in the testing notebook for prediction and evaluation.

## Dependencies

The following dependencies are required to run the notebooks:
- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib
- transformers (for Roberta model)

## Conclusion

This project demonstrates the application of machine learning models, specifically SVM, Naive Bayes and Logistic Regression, for sentiment analysis on the Twitter dataset. The performance of these models is compared with the ROBERTA model, showcasing the effectiveness of different approaches in sentiment analysis tasks.

Please refer to the project code and documentation for more details on the implementation and usage.


