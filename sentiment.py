import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

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

sentiments = [1, 0, 1, 0, 1, 0, 1, 0]

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text)  
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]  
    return ' '.join(stemmed_tokens)


processed_texts = [preprocess(text) for text in texts]


vectorizer = CountVectorizer()
features = vectorizer.fit_transform(processed_texts)


X_train, X_test, y_train, y_test = train_test_split(features, sentiments, test_size=0.5, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy on test set:", accuracy)


new_text = ["This product is amazing and works perfectly!"]
new_processed_text = [preprocess(new_text[0])]
new_features = vectorizer.transform(new_processed_text)
new_prediction = model.predict(new_features)
print("Predicted Sentiment for new text:", "Positive" if new_prediction[0] == 1 else "Negative")
