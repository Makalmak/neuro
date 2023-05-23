from flask import Flask, request, render_template, jsonify
from tensorflow import keras
from keras import layers
from keras import models

import numpy as np
import random
import io

import matplotlib.pyplot as plt

import h5py
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

app = Flask(__name__)

@app.route('/generate', methods=['POST', 'GET'])
def complete_text():

    result = ''

    if request.method == 'POST':
        i_text = request.form.get('text')

        # -----------Нейронка-----------------

        with open("movie_synopsis.csv", encoding="utf-8") as f:
            text = f.read().lower()

        def ok(x):
            return x.isalpha() | (x == ' ')

        text = "".join(c for c in text if ok(c))
        text = text[:100000]

        # создаем словари для кодирования и декодирования текста
        words = sorted(set(text.split()))
        print("Total words:", len(words))
        word_indices = dict((c, i) for i, c in enumerate(words))
        indices_word = dict((i, c) for i, c in enumerate(words))

        # Размер начальной последовательности можно отрегулировать
        maxlen = 10
        sentences = []
        next_words = []
        pointer = 0
        text_list = text.split()
        while pointer < len(words) - maxlen:
            sentences.append(text_list[pointer: pointer + maxlen])
            next_words.append(text_list[pointer + maxlen])
            pointer += 1

        x = np.zeros((len(sentences), maxlen, len(words)))
        y = np.zeros((len(sentences), len(words)))
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                x[i, t, word_indices[word]] = 1
            y[i, word_indices[next_words[i]]] = 1

        batch_size = 128

        def sample(preds, temperature=1.0):
            preds = np.asarray(preds).astype("float64")
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)



        model_new = models.load_model('model.hdf5')
        losses_new = []

        for diversity in [0.2, 0.5, 1.0, 1.2]:
          print("...Diversity:", diversity)

          generated = ""
          sentence = i_text.split()
          print('...Generating with seed: "' + " ".join(sentence) + '"')

          for i in range(100):
              x_pred = np.zeros((1, maxlen, len(words)))
              for t, word in enumerate(sentence):
                x_pred[0, t, word_indices[word]] = 1.0
              preds = model_new.predict(x_pred, verbose=0)[0]
              next_index = sample(preds, diversity)
              next_word = indices_word[next_index]
              sentence = sentence[1:]
              sentence.append(next_word)
              generated += next_word
              generated += " "
          print("...Generated: ", generated)
          print()
          history = model_new.fit(x, y, batch_size=batch_size, epochs=1)
          losses_new.append(history.history['loss'][0])

        result = generated

    # Отправка JSON
    return jsonify({
        'text': result
    })


if __name__ == '__main__':
    app.run()
