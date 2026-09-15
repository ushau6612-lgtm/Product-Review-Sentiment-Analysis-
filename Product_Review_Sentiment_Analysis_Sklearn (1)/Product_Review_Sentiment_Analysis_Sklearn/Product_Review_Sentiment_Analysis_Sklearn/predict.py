import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"

model = joblib.load(MODEL_PATH)

print("Product Review Sentiment Analysis")
print("-" * 40)

review = input("Enter a product review: ").strip()

if not review:
    print("Please enter a review.")
else:
    prediction = model.predict([review])[0]
    probabilities = model.predict_proba([review])[0]
    confidence = max(probabilities)

    print("\nPredicted Sentiment:", prediction.upper())
    print(f"Confidence: {confidence:.2%}")
