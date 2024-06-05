# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score

# Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display the first few rows of the dataset
print(train_data.head())

# Preprocess the dataset
# Handle Missing Values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to a large number of missing values
train_data.drop(columns=['Cabin'], inplace=True)

# Remove Duplicate Data
train_data.drop_duplicates(inplace=True)

# Feature and target separation
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

# Feature Scaling and Categorical Data Encoding using Pipelines
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model Experimentation
# Define the models
models = {
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC()
}

# Experiment with different hyperparameters
param_grid_rf = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30]
}

param_grid_svm = {
    'model__C': [0.1, 1, 10],
    'model__gamma': [1, 0.1, 0.01]
}

# Evaluation Function
def evaluate_model(model, param_grid):
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
    
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    y_pred = cross_val_predict(best_model, X, y, cv=5)
    
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    
    return best_model, precision, recall

# Evaluate RandomForest
best_rf, precision_rf, recall_rf = evaluate_model(models['RandomForest'], param_grid_rf)
print(f'RandomForest - Precision: {precision_rf}, Recall: {recall_rf}')

# Evaluate SVM
best_svm, precision_svm, recall_svm = evaluate_model(models['SVM'], param_grid_svm)
print(f'SVM - Precision: {precision_svm}, Recall: {recall_svm}')

# Model Evaluation
if precision_rf > precision_svm and recall_rf > recall_svm:
    best_model = best_rf
    print("Best Model: RandomForest")
else:
    best_model = best_svm
    print("Best Model: SVM")

# Bonus: Advanced Techniques
# Cross-Validation and Hyperparameter Tuning
cross_val_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cross_val_scores.mean()}')

# Final Model Training on Full Dataset
best_model.fit(X, y)

# If you want to save the model for later use
import joblib
joblib.dump(best_model, 'best_model.pkl')

