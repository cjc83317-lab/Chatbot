import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Preprocess text
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    filtered = [word for word in lemmatized if word not in stop_words]
    return ' '.join(filtered)

# Prepare training data
documents = []
labels = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        documents.append(preprocess(pattern))
        labels.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Vectorize text
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(documents)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer, label_encoder), f)

print("Model trained and saved.")