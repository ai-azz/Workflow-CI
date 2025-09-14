import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=42)
parser.add_argument('--max_features', type=int, default=5000)
parser.add_argument('--data_path', type=str, default=None)  # optional
args = parser.parse_args()

# mlflow config
if "MLFLOW_TRACKING_URI" in os.environ and os.environ["MLFLOW_TRACKING_URI"]:
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
else:
    tracking_uri = "file:mlruns"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("20newsgroups_experiment")
mlflow.sklearn.autolog()

script_dir = os.path.dirname(os.path.abspath(__file__))
if args.data_path:
    csv_path = args.data_path
else:
    candidate = os.path.join(script_dir, "20newsgroups_preprocessed.csv")
    candidate2 = os.path.join(script_dir, "newsgroups_preprocessing", "20newsgroups_preprocessed.csv")
    if os.path.isfile(candidate):
        csv_path = candidate
    elif os.path.isfile(candidate2):
        csv_path = candidate2
    else:
        raise FileNotFoundError(
            f"Dataset not found. Looked for:\n - {candidate}\n - {candidate2}\n"
        )

df = pd.read_csv(csv_path)
df['clean_text'] = df['clean_text'].fillna('')

# tain/test split
X = df['clean_text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
)

# vectorize & traim
vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1,2), min_df=5, max_df=0.8)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=args.random_state)
model.fit(X_train_tfidf, y_train)

# predict & eval
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)
print(classification_report(y_test, y_pred))

# save local artifacts
os.makedirs("artifacts", exist_ok=True)
joblib.dump(vectorizer, "artifacts/vectorizer.pkl")
joblib.dump(model, "artifacts/model.pkl")
print("Saved local artifacts to ./artifacts (vectorizer.pkl, model.pkl)")