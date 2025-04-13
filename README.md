# Offensive-Language-Classification

## Project Overview
This project aims to develop machine learning models to classify toxic content in online feedback. The primary objective is to predict whether a given comment is toxic or not (binary classification). The dataset includes fine-grained labels such as abusive, vulgar, menace, offense, and bigotry, which are used to enhance the model's learning. However, the final evaluation is based solely on the binary toxic label.

## Use for project implements:
Baseline Models: Logistic Regression and Random Forest.
Advanced Models: LSTM and GRU with pre-trained GloVe embeddings.
Transformer-Based Model: XLM-RoBERTa for multilingual toxic content classification.

## Dataset Description
The dataset consists of three filestrain.csv Labeled training data with the following columns id: Unique identifier for each comment.feedback_text: The text of the feedback to be classified. toxic: Binary label indicating if the comment is toxic (1 = toxic, 0 = not toxic). Additional fine-grained labels: abusive, vulgar, menace, offense, bigotry.
val.csv: Validation dataset with multilingual content and the toxic label.
test.csv: Unlabeled test dataset with multilingual content.

## Model Implementation Details
1. Exploratory Data Analysis (Label Distribution, Sentence Structure, Language Distribution, Common Words, word cloude, Missing Value)
2. Text Preprocessing(Cleaning,Stopword Removal,Stemming,Feature Extraction)
3. Baseline Models
    (Logistic Regression & Random Forest)
4. Advanced Models:
  ( LSTM and GRU )
5. Transformer-Based Model
     ( XLM-RoBERTa )

# Prediction Function
 A function predict_toxicity was implemented to make predictions using all models for a given input sentence. Example:
  
Input Sentence: You are the worst person I have ever met.

Toxic: Prediction = 0, Probability = 0.2651
Abusive: Prediction = 0, Probability = 0.0242
Vulgar: Prediction = 0, Probability = 0.4475
Menace: Prediction = 0, Probability = 0.0490
Offense: Prediction = 0, Probability = 0.4968
Bigotry: Prediction = 0, Probability = 0.1015
 

## Model Evaluation Results
The transformer-based model (XLM-RoBERTa) outperformed all other models, achieving the highest AUC-ROC score of 0.8722.

precision    recall   f1-score    support

   Non-Toxic       0.85      1.00      0.92       706
       Toxic       0.79      0.08      0.15       134

accuracy                           0.85       840
   macro avg       0.82      0.54      0.53       840
weighted avg       0.84      0.85      0.80       840


## Generate predictions on test set than  Convert predictions to binary than predictions for the test dataset (i have predictade  predicted_toxic. CSV  test dataset )

## Additional Observations
1. Class Imbalance:
The dataset is heavily imbalanced, with the toxic class being underrepresented.
Techniques like oversampling (e.g., SMOTE) or class-weight adjustments could improve recall for the toxic class.

2. Multilingual Content:
The validation and test datasets include non-English content, but the models were trained primarily on English data.
Using multilingual embeddings ( XLM-R) significantly improved performance on multilingual data.

3. Future Improvements:
Train for more epochs with early stopping to prevent overfitting.
Experiment with additional transformer-based models like BERT or RoBERTa for better results.
     
