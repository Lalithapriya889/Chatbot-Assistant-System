import random
import nltk
import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Intents data (you can also load this from a separate file if preferred)
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good evening"],
            "responses": ["Hello!", "Hi there!", "Greetings!", "How can I help you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Bye!", "See you soon!", "Take care!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful"],
            "responses": ["You're welcome!", "No problem!", "Any time!"]
        },
        {
            "tag": "hours",
            "patterns": ["What are your hours?", "When are you open?", "Opening hours?"],
            "responses": ["We're open from 9am to 5pm, Monday to Friday."]
        }
    ]
}

def train_chatbot():
    corpus = []
    tags = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            corpus.append(pattern)
            tags.append(intent["tag"])

    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    X = vectorizer.fit_transform(corpus)

    with open("chatbot_model.pkl", "wb") as f:
        pickle.dump((vectorizer, X, tags, intents), f)

    print("Model trained and saved to 'chatbot_model.pkl'.")

def load_model():
    if not os.path.exists("chatbot_model.pkl"):
        train_chatbot()

    with open("chatbot_model.pkl", "rb") as f:
        return pickle.load(f)

def predict_class(msg, vectorizer, X, tags):
    vec = vectorizer.transform([msg])
    sims = cosine_similarity(vec, X)
    idx = sims.argmax()
    return tags[idx]

def get_response(tag, intents):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

if _name_ == "_main_":
    vectorizer, X, tags, intents = load_model()

    print("🤖 Chatbot Assistant is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break
        tag = predict_class(user_input, vectorizer, X, tags)
        response = get_response(tag, intents)
        print("Bot:", response)
