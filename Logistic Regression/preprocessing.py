import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def LoadData():
    """
    Loads the Titanic dataset from the 'data' folder.
    Expects: train.csv, test.csv, and gender_submission.csv
    """
    train = pd.read_csv('./Logistic Regression/data/train.csv')
    test = pd.read_csv('./Logistic Regression/data/test.csv')
    y_test = pd.read_csv('./Logistic Regression/data/gender_submission.csv')
    return train, test, y_test

def EncodeCategorical(train, test):
    """
    Converts text-based categories into numbers.
    Logistic Regression requires numerical input to perform matrix multiplication.
    - Sex: 'male'/'female' -> 1/0
    - Embarked: 'S'/'C'/'Q' -> 2/0/1
    """
    le = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        # We fit on train and transform both to ensure the mapping is consistent
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])
    return train, test

def ScaleFeatures(train, test):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Formula: z = (x - u) / s
    This is crucial for Gradient Descent to converge quickly and prevent
    high-magnitude features (like Fare) from dominating the weights.
    """
    scaler = StandardScaler()
    features_to_scale = ['Age', 'Fare']
    train[features_to_scale] = scaler.fit_transform(train[features_to_scale])
    test[features_to_scale] = scaler.transform(test[features_to_scale])
    return train, test
    
def PreprocessData(train, test, y_test):
    """
    The main cleaning pipeline:
    1. Merges labels
    2. Handles missing values via 'Title' extraction
    3. Removes noise (Cabin, PassengerId, etc.)
    4. Encodes and Scales data
    """
    # Merge y_test with test data so labels and features stay aligned after cleaning
    test = test.merge(y_test, on='PassengerId', how='left')

    # Drop 'Cabin' immediately as it has ~77% missing values
    train.drop(columns=['Cabin'], inplace=True)
    test.drop(columns=['Cabin'], inplace=True)

    # 1. Feature Engineering: Extract Title (Mr, Mrs, Master) from Name
    # This helps us guess the missing ages more accurately.
    train['Title'] = train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    test['Title'] = test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # 2. Impute Missing Ages
    # Instead of a global average, we use the median age of people with the same title.
    title_age_median = train.groupby('Title')['Age'].median()
    train['Age'] = train['Age'].fillna(train['Title'].map(title_age_median))
    
    # Use training medians for test set to avoid 'Data Leakage'
    test['Age'] = test['Age'].fillna(test['Title'].map(title_age_median))

    # 3. Clean remaining nulls
    # Embarked has very few missing values; we drop those specific rows.
    train.dropna(subset=['Embarked'], inplace=True)
    test.dropna(subset=['Embarked', 'Age', 'Fare'], inplace=True)

    # 4. Drop redundant columns
    # These are unique identifiers or text that don't help the model generalize.
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Title']
    train.drop(columns=columns_to_drop, inplace=True)
    test.drop(columns=columns_to_drop, inplace=True)

    # 5. Transform data for the Model
    train, test = EncodeCategorical(train, test)
    train, test = ScaleFeatures(train, test)

    return train, test
