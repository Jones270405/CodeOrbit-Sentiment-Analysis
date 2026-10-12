# Sentiment Analysis on Text Data

A simple NLP project that classifies input text as positive or negative using TF-IDF features and Logistic Regression.

## Requirements
- Python 3.10+
- scikit-learn

## Run
```bash
python -m pip install -r requirements.txt
python sentiment_analysis.py
```

Type a sentence and press Enter. Type `quit` to exit.

## How it works
1. Use the small example dataset embedded in the program.
2. Convert text to TF-IDF features, including unigrams and bigrams.
3. Train a Logistic Regression classifier.
4. Predict sentiment for user-entered text and display the model's probability estimate.

## Limitations
The training dataset is deliberately small and only demonstrates the workflow. Its predictions and confidence values are not reliable for real-world use. A portfolio version should use a larger, documented dataset, a held-out test set, metrics, and error analysis.
