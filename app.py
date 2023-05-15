from flask import Flask, request, jsonify
from transformers import pipeline
from config import *

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=["GET"])
def health_check():
    """Confirms service is running"""
    return "Sentiment Analysis service is up and running."

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']

    sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)
    language_detection_model = pipeline("text-classification", model=language_detection_model_path, tokenizer=sentiment_model_path)
    spam_model = pipeline("text-classification", model=spam_model_path, tokenizer=spam_model_path)
    toxicity_model = pipeline("text-classification", model=toxicity_model_path, tokenizer=toxicity_model_path)
    fake_news_model = pipeline("text-classification", model=fake_news_model_path, tokenizer=fake_news_model_path)

    sentiment_pred = sentiment_model(text)[0]
    language_detection_pred = language_detection_model(text)[0]
    spam_pred = spam_model(text)[0]
    toxicity_pred = toxicity_model(text)[0]
    fake_news_pred = fake_news_model(text)[0]

    output_data = {sentiment_model_path.split('/')[0] :sentiment_pred,
                   language_detection_model_path.split('/')[0]: language_detection_pred,
                   fake_news_model_path.split('/')[0]: spam_pred,
                   toxicity_model_path.split('/')[0]: toxicity_pred,
                   spam_model_path.split('/')[0]: fake_news_pred}
    return jsonify(output_data)

app.run(host="0.0.0.0")



