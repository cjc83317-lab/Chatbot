import json
import pickle
import random
import requests
import nltk
import re
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load resources
with open('intents.json') as file:
    intents = json.load(file)

with open('chatbot_model.pkl', 'rb') as f:
    model, vectorizer, label_encoder = pickle.load(f)

# Preprocess input
def preprocess(text):
    from nltk.tokenize import wordpunct_tokenize
    tokens = wordpunct_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    filtered = [word for word in lemmatized if word not in stop_words]
    return ' '.join(filtered)

# Predict intent
def predict_class(sentence):
    bow = vectorizer.transform([preprocess(sentence)])
    probs = model.predict_proba(bow)[0]
    ERROR_THRESHOLD = 0.5
    results = [(i, p) for i, p in enumerate(probs) if p > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    print("Predicted intents:", [(label_encoder.classes_[i], p) for i, p in results])

    if not results:
        return [{'intent': 'fallback', 'probability': '1.0'}]
    return [{'intent': label_encoder.classes_[results[0][0]], 'probability': str(results[0][1])}]


# Fallback search
def search_web_for_answer(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    try:
        response = requests.get(url).json()
        abstract = response.get("AbstractText")
        related = response.get("RelatedTopics", [])
        if abstract:
            return abstract
        elif related:
            return related[0].get("Text", "I found something, but it's not very detailed.")
        else:
            return f"I couldn't find a clear answer, but you can check online: https://duckduckgo.com/?q={query.replace(' ', '+')}"
    except Exception as e:
        return f"Search error: {e}"

# Math detection and solving
def is_math_query(text):
    return bool(re.search(r'[\d\+\-\*/\^=]', text))

def solve_math(text):
    try:
        expression = re.sub(r'[^\d\+\-\*/\.\(\)]', '', text)
        result = eval(expression)
        return f"The answer is {result}"
    except Exception as e:
        return f"Sorry, I couldn't solve that. Error: {e}"


# Get response
def get_response(intents_list, intents_json, user_input):
    tag = intents_list[0]['intent']

    if tag == 'math' or is_math_query(user_input):
        return solve_math(user_input)

    if tag == 'fallback':
        return search_web_for_answer(user_input)

    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Run chatbot
print("GO! Bot is running!")

while True:
    message = input("You: ")
    intents_list = predict_class(message)
    response = get_response(intents_list, intents, message)
    print(f"Bot ({intents_list[0]['intent']}): {response}")