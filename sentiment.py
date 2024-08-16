import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample data
texts = [
    "I love this product",
    "I hate this product",
    "This is the best thing ever!",
    "This is the worst experience I've had",
    "Amazing product, very happy with it",
    "Terrible product, completely useless",
    "I'm extremely satisfied",
    "I'm really disappointed",
]

# Corresponding sentiment labels (1 for positive, 0 for negative)
sentiments = [1, 0, 1, 0, 1, 0, 1, 0]

# Preprocessing: Tokenization, Stop Words Removal, and Stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text)  # Tokenization
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]  # Stop Words Removal
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]  # Stemming
    return ' '.join(stemmed_tokens)

# Preprocess all texts
processed_texts = [preprocess(text) for text in texts]

# Feature Extraction: Convert text to features using Bag of Words
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(processed_texts)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, sentiments, test_size=0.5, random_state=42)

# Model Training: Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Model Evaluation: Predict and check accuracy on the test set
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy on test set:", accuracy)

# Sentiment Prediction: Predict sentiment for new, unseen text
new_text = ["This product is amazing and works perfectly!"]
new_processed_text = [preprocess(new_text[0])]
new_features = vectorizer.transform(new_processed_text)
new_prediction = model.predict(new_features)
print("Predicted Sentiment for new text:", "Positive" if new_prediction[0] == 1 else "Negative")
