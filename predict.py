from flask import Flask, jsonify, request
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
import emoji
import pandas as pd
import numpy as np
import os
import io
import re
import nltk
import json
nltk.download('punkt')

app = Flask(__name__)


@app.route("/descam/predict", methods=["POST"])
def run_app():
    data = {"success": False, "output": []}
    try:
        params = request.get_json()
        print(params)
        if params is None:
            return jsonify(data)
        if params:
            output = run_prediction(params)

            print(output)

            data["success"] = True
            data["output"] = output

    except:
        print("Get exception")
    return jsonify(data)


def preprocessing(data, tokenizer):
    vocab_size = 50000
    max_length = 500
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    input = data

    input = input.lower()  # lowercase
# ilangin tab
    temp = input
    temp = re.sub(r'\n', ' ', temp)
# Ilangin angka and simbol
    temp = re.sub('[^a-zA-Z,.?!]+', ' ', temp)
# Ilangin Link
    temp = re.sub(
        r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", "", temp)
# Hapus Emoji dsb
    temp = emoji.demojize(temp)
    temp = re.sub(':[A-Za-z_-]+:', ' ', temp)
# hapus hashtag
    temp = re.sub(r'#(\S+)', r'\1', temp)
# rapihin spasi
    temp = re.sub('[ ]+', ' ', temp)
    temp = ' '.join(
        re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", temp).split())
    input = temp

    tokenizer = tokenizer
    input = [input]
    input_sequence = np.array(tokenizer.texts_to_sequences(input))
    input_sequence = pad_sequences(
        input_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return input_sequence


def predict(input, model):
    output = ""
    model = model
    pred = model.predict(input)

    if(pred > 0.4):
        output = "Legal"
    else:
        output = "Ilegal"

    return output


def run_prediction(input):
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    pre = preprocessing(input, tokenizer)
    prediction = predict(pre, model)

    return prediction


model = tf.keras.models.load_model("model1.h5")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=int(os.environ.get("PORT", 8080)))
