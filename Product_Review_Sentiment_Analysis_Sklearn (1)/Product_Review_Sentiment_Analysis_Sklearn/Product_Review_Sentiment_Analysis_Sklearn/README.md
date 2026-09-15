# Product Review Sentiment Analysis using Python + Scikit-learn

## Project Overview
This machine learning project classifies product reviews into three sentiment categories:

- Positive
- Negative
- Neutral

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib
- Matplotlib

## How the Project Works
1. Product review text is converted into numerical features using TF-IDF.
2. Logistic Regression learns sentiment patterns.
3. The trained model predicts the sentiment of a new review.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```

This creates:
- `sentiment_model.pkl`
- `sentiment_distribution.png`

### 3. Predict sentiment
```bash
python predict.py
```

Enter a product review when prompted.

## Dataset
The included dataset is synthetic and intended for learning and demonstration.

## Example
Input:
```text
The product quality is excellent and delivery was fast
```

Output:
```text
Predicted Sentiment: POSITIVE
```

## Important Note
This project is an educational demonstration. For production use, train and validate the model using a larger, real-world, properly labeled dataset.
