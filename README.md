# SMS Fraud Detection System

A machine learning system for detecting fraudulent SMS messages using natural language processing and classification algorithms.

## Features

- **Text Preprocessing**: Cleans and normalizes SMS text data
- **Feature Extraction**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Classification Models**: Implements both Naive Bayes and Logistic Regression classifiers
- **Performance Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Interactive Interface**: User-friendly command-line interface for real-time SMS classification
- **Model Persistence**: Save and load trained models for future use

## Files

- `simple_sms_detector.py` - Main SMS fraud detection implementation
- `demo_sms_detector.py` - Demonstration script showing the system in action
- `sms_fraud_detector.pkl` - Saved trained model (generated after first run)
- `requirements.txt` - Python dependencies (for environments with package managers)

## Quick Start

### Running the Demo

```bash
python demo_sms_detector.py
```

This will:
1. Load or train the models
2. Test with sample legitimate and fraudulent messages
3. Display classification results with confidence scores

### Interactive Classification

```bash
python simple_sms_detector.py
```

This provides an interactive interface where you can:
- Enter SMS messages to classify
- Choose between Naive Bayes and Logistic Regression models
- Get detailed predictions with confidence scores

## Model Performance

Based on testing with a balanced dataset of 40 SMS messages:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 87.5% | 100% | 87.5% | 0.93 |
| Logistic Regression | 75% | 100% | 75% | 0.86 |

## Text Preprocessing Steps

1. **Lowercase Conversion**: Convert all text to lowercase
2. **URL Removal**: Strip out hyperlinks and URLs
3. **Phone Number Removal**: Remove phone number patterns
4. **Punctuation Removal**: Clean punctuation marks
5. **Tokenization**: Split text into individual words
6. **Stopword Removal**: Remove common English stopwords
7. **Lemmatization**: Reduce words to their base forms

## Feature Extraction

The system uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features:

- **Term Frequency (TF)**: How often a word appears in a message
- **Inverse Document Frequency (IDF)**: How unique a word is across all messages
- **TF-IDF Score**: TF × IDF, giving higher weight to important, unique words

## Classification Models

### Naive Bayes
- Probabilistic classifier based on Bayes' theorem
- Assumes independence between features
- Fast training and prediction
- Works well with text classification tasks

### Logistic Regression
- Linear model for binary classification
- Uses sigmoid function to predict probabilities
- Trained using gradient descent optimization
- Provides interpretable feature weights

## Usage Examples

### Classify a Single Message

```python
from simple_sms_detector import SimpleSMSFraudDetector

detector = SimpleSMSFraudDetector()
detector.load_model('simple_sms_detector.pkl')

message = "URGENT: Your account has been suspended. Click here to verify."
result = detector.predict_sms(message, model_name='Naive Bayes')

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Training a New Model

```python
detector = SimpleSMSFraudDetector()

# Create your dataset (list of tuples: (message, label))
dataset = [
    ("Hello, how are you?", 0),  # 0 = legitimate
    ("WIN $1000 NOW! Click here!", 1),  # 1 = fraudulent
    # ... more messages
]

# Train the model
detector.build_vocabulary(dataset)
X, y = detector.compute_tf_idf(dataset)

# Split data and train
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]

nb_model = detector.train_naive_bayes(X_train, y_train)
detector.models['Naive Bayes'] = nb_model
```

## Dependencies

The system is designed to work with basic Python libraries only:
- `re` - Regular expressions for text cleaning
- `string` - String operations
- `collections` - Counter for word frequencies
- `math` - Mathematical functions for calculations
- `pickle` - Model serialization

For environments with package managers, the following are recommended:
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- nltk >= 3.7

## Dataset

The system includes a sample dataset with:
- 20 legitimate SMS messages (normal conversations, notifications, etc.)
- 20 fraudulent SMS messages (scams, phishing attempts, fake offers)

The dataset covers common fraud patterns:
- Urgent account alerts
- Lottery/giveaway scams
- Investment opportunities
- Phishing links
- Fake security warnings

## Future Enhancements

- Support for additional classification algorithms (SVM, Random Forest)
- Integration with real SMS APIs
- Web-based user interface
- Multi-language support
- Advanced feature engineering (sentiment analysis, message length, etc.)
- Real-time model updating with user feedback

## License

This project is open source and available for educational and research purposes.
