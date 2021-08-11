from flask import Flask, render_template, request
import pickle
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    model = load_model('model.h5')

    if request.method == 'POST':
        namequery = request.form['text']
        data = [namequery]

        x_seq = tokenizer.texts_to_sequences(data)
        encoded = pad_sequences(x_seq, maxlen=1500, padding='post')

        prediction = model.predict(encoded)

        if prediction[0] >= 0.0:
            sentiment = 'Positive'
        else:
            sentiment = 'Negative'

        K.clear_session()
    return render_template('index.html', prediction_text=sentiment)

    # K.clear_session()


if __name__ == '__main__':
    app.run(debug=True)
