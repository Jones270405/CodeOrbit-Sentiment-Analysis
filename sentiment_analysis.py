"""
Simple sentiment analysis using TF-IDF and Logistic Regression.
Run: python sentiment_analysis.py
This educational demo trains on a small example dataset included in the file.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TRAINING_DATA = [
    ("I love this product", "positive"),
    ("This is wonderful and useful", "positive"),
    ("The experience was excellent", "positive"),
    ("I am very happy with the service", "positive"),
    ("Amazing quality and fast delivery", "positive"),
    ("This works really well", "positive"),
    ("I enjoyed every moment", "positive"),
    ("The app is easy to use", "positive"),
    ("I hate this product", "negative"),
    ("This is terrible and useless", "negative"),
    ("The experience was awful", "negative"),
    ("I am very disappointed with the service", "negative"),
    ("Poor quality and late delivery", "negative"),
    ("This does not work at all", "negative"),
    ("I regret buying this", "negative"),
    ("The app is confusing and frustrating", "negative"),
]

def build_model():
    texts = [text for text, label in TRAINING_DATA]
    labels = [label for text, label in TRAINING_DATA]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(texts, labels)
    return model

def main():
    model = build_model()
    print("=== Simple Sentiment Analysis ===")
    print("Enter a sentence to classify as positive or negative. Type 'quit' to exit.")
    while True:
        text = input("\nText: ").strip()
        if text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not text:
            print("Please enter some text.")
            continue
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        confidence = max(probabilities) * 100
        print(f"Predicted sentiment: {prediction.upper()}")
        print(f"Model confidence: {confidence:.1f}%")

if __name__ == "__main__":
    main()
