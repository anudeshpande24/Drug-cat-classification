from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
from transformers import BertTokenizer, BertModel

# Load the model
with open('drugcat.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    description = data['description']
    embedding = get_bert_embeddings(description)
    category = model.predict(embedding)
    return jsonify({'category': category[0]})

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run()
