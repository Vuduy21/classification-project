import os
import re
import joblib
import torch
import nltk
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Tải các tài nguyên NLTK
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ============ Tiền xử lý văn bản ============
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def load_vocab(vocab_file='models/vocab.txt'):
    vocab = {}
    if os.path.exists(vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                word = line.strip()
                if word:
                    vocab[word] = idx + 2  
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1
    else:
        print(f"Error: File {vocab_file} not found. BiLSTM will fail.")
        raise FileNotFoundError(f"Vocabulary file {vocab_file} not found.")
    return vocab

def encode_text(text, vocab, max_len=100):
    tokens = text.split()
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    ids = ids[:max_len] + [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

vocab = load_vocab()

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=2):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # [B, T, D]
        lstm_out, _ = self.lstm(embedded)  # [B, T, 2H]
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # [B, T]
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [B, 2H]
        logits = self.classifier(context)
        return logits

# Logistic Regression
logistic_model = joblib.load('models/logistic/logistic_regression_model.pkl')
logistic_vectorizer = joblib.load('models/logistic/tfidf_vectorizer.pkl')

# Naive Bayes
nb_model = joblib.load('models/naive_bayes/naive_bayes_model.pkl')
nb_vectorizer = joblib.load('models/naive_bayes/nb_count_vectorizer.pkl')

# Random Forest
rf_model = joblib.load('models/randomforest/randomforest.pkl')
rf_vectorizer = joblib.load('models/randomforest/random_tfidf_vectorizer.pkl')

# SVM
svm_model = joblib.load('models/svm/svm.pkl')
svm_vectorizer = joblib.load('models/svm/svm_tfidf_vectorizer.pkl')

# LSTM (Keras)
lstm_model = load_model('models/lstm/lstm.keras')
with open('models/lstm/lstm_tokenizer.pkl', 'rb') as handle:
    lstm_tokenizer = pickle.load(handle)
MAXLEN = 150

# BiLSTM-Attention (PyTorch)
with torch.serialization.safe_globals([BiLSTMAttention]):
    bilstm_model = torch.load('models/bilstm_attention_full.pt', map_location=torch.device('cpu'), weights_only=False)
bilstm_model.eval()
class BiLSTMConfig:
    max_len = 100
bilstm_config = BiLSTMConfig()

# DistilBERT
distil_tokenizer = AutoTokenizer.from_pretrained("models/distilBERT")
distil_model = AutoModelForSequenceClassification.from_pretrained("models/distilBERT")

# DeBERTa
deberta_tokenizer = AutoTokenizer.from_pretrained("models/deberta")
deberta_model = AutoModelForSequenceClassification.from_pretrained("models/deberta")

def predict_logistic(text):
    clean = preprocess(text)
    vec = logistic_vectorizer.transform([clean])
    return logistic_model.predict(vec)[0]

def predict_naive_bayes(text):
    clean = preprocess(text)
    vec = nb_vectorizer.transform([clean])
    return nb_model.predict(vec)[0]

def predict_randomforest(text):
    clean = preprocess(text)
    vec = rf_vectorizer.transform([clean])
    return rf_model.predict(vec)[0]

def predict_svm(text):
    clean = preprocess(text)
    vec = svm_vectorizer.transform([clean])
    return svm_model.predict(vec)[0]

def predict_lstm(text):
    clean = preprocess(text)
    seq = lstm_tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding='post', truncating='post')
    pred = lstm_model.predict(padded).flatten()
    return int(pred > 0.5)

def predict_bilstm(text):
    clean = preprocess(text)
    encoded = encode_text(clean, vocab, max_len=bilstm_config.max_len)
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        output = bilstm_model(input_tensor)
        probs = torch.softmax(output, dim=1)
    return int(torch.argmax(probs, dim=1).item())

def predict_distilbert(text):
    inputs = distil_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = distil_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    return int(torch.argmax(probs).item())

def predict_deberta(text):
    inputs = deberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = deberta_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    return int(torch.argmax(probs).item())

predict_functions = {
    "logistic": predict_logistic,
    "naive_bayes": predict_naive_bayes,
    "random_forest": predict_randomforest,
    "svm": predict_svm,
    "lstm": predict_lstm,
    "bilstm": predict_bilstm,
    "distilbert": predict_distilbert,
    "deberta": predict_deberta
}

# ============ API Endpoint ============
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  
    if not data or 'texts' not in data:
        return jsonify({"error": "No texts provided"}), 400
    
    texts = data['texts']  
    if not isinstance(texts, list):
        return jsonify({"error": "Expected a list of texts"}), 400
    
    models = data.get('models', list(predict_functions.keys()))
    if not isinstance(models, list):
        return jsonify({"error": "Expected a list of models"}), 400
    
    valid_models = [model for model in models if model in predict_functions]
    if not valid_models:
        return jsonify({"error": "No valid models selected"}), 400
    
    try:
        results = []
        for text in texts:
            result = {}
            for model in valid_models:
                result[model] = int(predict_functions[model](text))
            results.append(result)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)