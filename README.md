# product-Review-Sentiment-Analysis-
Product Review Sentiment Analysis is a Natural Language Processing (NLP) and machine learning project that automatically analyzes customer product reviews and classifies them into three sentiment categories: Positive, Negative, or Neutral
# Product Review Sentiment Analysis using Python and Scikit-learn

## Project Overview

Product Review Sentiment Analysis is a Natural Language Processing and machine learning project designed to classify customer product reviews into three sentiment categories:

* Positive
* Negative
* Neutral

The project uses TF-IDF Vectorization to transform review text into numerical features and Logistic Regression to classify the sentiment of customer feedback.

A trained machine learning model is saved using Joblib and can be used to predict the sentiment of new product reviews. The project also includes a Streamlit-based web application for interactive sentiment analysis.

## Objectives

The main objectives of this project are:

* Analyze customer product reviews automatically.
* Identify whether a review expresses positive, negative, or neutral sentiment.
* Apply Natural Language Processing techniques to textual data.
* Train and evaluate a machine learning classification model.
* Provide an interactive interface for sentiment prediction.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* Joblib
* Matplotlib
* Streamlit

## Machine Learning Workflow

The project follows these steps:

1. Load the product review dataset.
2. Separate review text and sentiment labels.
3. Split the dataset into training and testing sets.
4. Convert review text into numerical features using TF-IDF.
5. Train a Logistic Regression classification model.
6. Evaluate the model using accuracy and classification metrics.
7. Save the trained model as a `.pkl` file.
8. Use the saved model to predict sentiment for new reviews.
9. Visualize the distribution of sentiments.

## Dataset

The project contains a product review dataset stored at:

```text
data/product_reviews.csv
```

The dataset contains review text and corresponding sentiment labels.

The sentiment classes are:

```text
positive
negative
neutral
```

The included dataset is synthetic and intended for educational and demonstration purposes.

## Model

The project uses a Scikit-learn Pipeline containing:

```text
TF-IDF Vectorizer
        |
        v
Logistic Regression
        |
        v
Sentiment Prediction
```

### TF-IDF Vectorizer

TF-IDF converts textual reviews into numerical representations based on the importance of words within the dataset.

### Logistic Regression

Logistic Regression is used as the classification algorithm to identify sentiment patterns and predict one of the three sentiment classes.

## Project Structure

```text
Product_Review_Sentiment_Analysis_Sklearn/
│
├── data/
│   └── product_reviews.csv
│
├── train_model.py
├── predict.py
├── app.py
├── sentiment_model.pkl
├── sentiment_distribution.png
├── requirements.txt
├── README.md
├── HOW_TO_RUN.md
└── run_frontend.bat
```

## Installation

Make sure Python is installed on your system.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Train the Model

Run the following command:

```bash
python train_model.py
```

The training script:

* Loads the dataset.
* Splits the data into training and testing sets.
* Creates a TF-IDF and Logistic Regression pipeline.
* Trains the model.
* Calculates model accuracy.
* Displays a classification report.
* Saves the trained model.
* Generates a sentiment distribution chart.

The trained model will be saved as:

```text
sentiment_model.pkl
```

## Predict Sentiment from the Terminal

Run:

```bash
python predict.py
```

Enter a product review when prompted.

Example:

```text
Enter a product review: Great quality and excellent performance
```

Example output:

```text
Predicted Sentiment: POSITIVE
Confidence: 95.20%
```

## Run the Streamlit Application

The project includes an interactive Streamlit web application.

Run:

```bash
streamlit run app.py
```

The application provides:

* Single review sentiment analysis
* Sentiment prediction confidence
* Pre-written sample reviews
* Sentiment distribution visualization
* Review dataset viewing

## Example

### Input

```text
This product is excellent and works perfectly.
```

### Output

```text
Predicted Sentiment: POSITIVE
```

Another example:

### Input

```text
Not worth the price and stopped working after two days.
```

### Output

```text
Predicted Sentiment: NEGATIVE
```

## Model Evaluation

The model is evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report

The evaluation results are generated automatically when running:

```bash
python train_model.py
```

## Visualization

The project generates a sentiment distribution chart:

```text
sentiment_distribution.png
```

This visualization shows the number of reviews belonging to each sentiment category.

## Use Cases

This type of sentiment analysis system can be useful for:

* E-commerce review analysis
* Customer feedback monitoring
* Product quality analysis
* Customer satisfaction measurement
* Brand reputation analysis
* Automated review classification
* Business intelligence applications

## Limitations

This project is intended primarily for learning and demonstration.

The included dataset is synthetic, so the model may not perform reliably on complex real-world customer reviews. Real-world deployment would require a larger and more diverse labeled dataset, additional text preprocessing, and thorough model validation.

## Future Improvements

Possible improvements include:

* Training with a larger real-world dataset.
* Adding advanced text preprocessing.
* Comparing Logistic Regression with other classification algorithms.
* Using n-grams and optimized TF-IDF parameters.
* Adding deep learning or transformer-based models.
* Adding multilingual sentiment analysis.
* Deploying the application to a cloud platform.
* Adding batch CSV upload and sentiment prediction.
* Providing detailed sentiment analytics and dashboards.

## License

This project is intended for educational and demonstration purposes.
