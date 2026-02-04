import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def LoadPretrainedModel():
    """
    Initializes a Logistic Regression model instance.
    You can swap this for your custom class to test your implementation!
    """
    model = LogisticRegression()
    return model

def train_model(model, x_train, y_train):
    """
    Fits the model to the training data.
    This finds the relationship between features (like Sex, Age) 
    and the target (Survived).
    """
    model.fit(x_train, y_train)
    return model

def predict(model, x_test):
    """
    Generates class predictions (0 or 1) for new, unseen data.
    """
    return model.predict(x_test)

def evaluate_model(model, x_test, y_test):
    """
    Calculates various performance metrics to determine how 'smart' 
    the model is.
    """
    # 1. Generate predictions first
    y_pred = predict(model, x_test)
    
    # 2. Accuracy: How many were correct overall?
    accuracy = accuracy_score(y_test, y_pred)
    
    # 3. Precision: Quality of the positive (Survived) predictions.
    # High precision means few people were wrongly predicted to survive.
    precision = precision_score(y_test, y_pred)
    
    # 4. Recall: Ability to find all actual survivors.
    # High recall means the model didn't miss many survivors.
    recall = recall_score(y_test, y_pred)
    
    # 5. F1-Score: A balance between Precision and Recall.
    # Useful when survival classes are not perfectly 50/50.
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1
