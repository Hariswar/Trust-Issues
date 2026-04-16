from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
from pathlib import Path
import numpy as np

# import preprocessing functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing import (
    clean_normalize_dataset, 
    split_dataset, 
    read_train, 
    read_test, 
    read_evaluation,
    read_dataset_two,
    dataset_two_train,
    dataset_two_test
)

# prepare data
train_data = clean_normalize_dataset(read_train)
test_data = clean_normalize_dataset(read_test)

# combine text and title for features
X_train = (train_data['title'] + ' ' + train_data['text']).values
y_train = train_data['label'].values

X_test = (test_data['title'] + ' ' + test_data['text']).values
y_test = test_data['label'].values

# vectorize text data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# define models
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(max_iter=2000, random_state=1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=1)
}

# train and evaluate each model
for model_name, model in models.items():
    # train and predict
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # print results
    print(f"\n{model_name}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # show confusion matrix
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {cm}")
