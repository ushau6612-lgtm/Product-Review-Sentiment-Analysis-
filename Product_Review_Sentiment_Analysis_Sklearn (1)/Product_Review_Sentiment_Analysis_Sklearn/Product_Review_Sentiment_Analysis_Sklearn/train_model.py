import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "product_reviews.csv"
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
CHART_PATH = BASE_DIR / "sentiment_distribution.png"

df = pd.read_csv(DATA_PATH)

X = df["review_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True)),
    ("classifier", LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

counts = df["sentiment"].value_counts()
counts.plot(kind="bar", title="Sentiment Distribution", figsize=(7, 5))
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(CHART_PATH)
print(f"Chart saved to: {CHART_PATH}")
