
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def run_all_models(df):
    df['label'] = df['label'].replace({'ddos': 1, 'normal': 0}).astype(int)
    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['label'], errors='ignore')
    y_true = df['label']

    results = {}

    # Isolation Forest (unsupervised)
    iso_clf = IsolationForest(contamination=0.05, random_state=42)
    iso_clf.fit(X)
    iso_pred = [0 if i == 1 else 1 for i in iso_clf.predict(X)]
    results['IsolationForest'] = {
        'accuracy': accuracy_score(y_true, iso_pred),
        'precision': precision_score(y_true, iso_pred, zero_division=0),
        'recall': recall_score(y_true, iso_pred, zero_division=0),
        'f1_score': f1_score(y_true, iso_pred, zero_division=0)
    }

    # Supervised Models
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

    if len(set(y_train)) > 1:
        classifiers = {
            'RandomForest': RandomForestClassifier(),
            'NaiveBayes': GaussianNB(),
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'SVM': SVC()
        }

        for name, model in classifiers.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0)
                }
            except:
                results[name] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}

    best_model = max(results, key=lambda k: results[k]['f1_score'])
    return results, best_model

def save_model_comparison(results):
    os.makedirs('outputs', exist_ok=True)
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('outputs/model_comparison.csv')

def save_anomalies(df):
    os.makedirs('outputs', exist_ok=True)
    df['label'] = df['label'].replace({'ddos': 1, 'normal': 0}).astype(int)
    flagged = df[df['label'] == 1]
    flagged.to_csv('outputs/flagged_anomalies.csv', index=False)
