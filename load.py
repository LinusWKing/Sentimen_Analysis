import os
import pickle
import re
import string

import nltk
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, models, optimizers

nltk.download('stopwords')
nltk.download('wordnet')
stop = stopwords.words('english')


def process_text(text):
    for punctuation in '&;':
        text = text.replace(punctuation, '')
    printable = set(string.printable)
    s = list(filter(lambda x: x in printable, text))
    processed_text = ''.join([str(e) for e in s])
    processed_text = '<data>\n' + processed_text + '</data>'
    return processed_text


def load_data():
    # traverse root directory, and list directories as dirs and files as files
    for root, dir, files in os.walk("."):
        for file in files:
            if file.endswith('.review') and not file.startswith('unlabeled'):
                df = parse_xml(os.path.join(root, file))

    return pd.concat(df)


tmp_df = []


def parse_xml(path):
    with open(path, 'r') as file:
        text = file.read()
        text = process_text(text)
        df = pd.read_xml(text)
        tmp_df.append(df)
    file.close()

    return tmp_df


df = load_data()


def process_df(df):

    cols = ['helpful', 'rating', 'review_text']
    dropped_columns = [col for col in df.columns if col not in cols]

    for col in dropped_columns:
        df.drop([col], axis=1, inplace=True)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    helpful_list = df['helpful'].tolist()
    new_helpful_list = []
    for i in range(len(helpful_list)):
        s = helpful_list[i].split('of')
        first = int(s[0])
        last = int(s[-1])
        new_helpful = round(first / last, 2)
        new_helpful_list.append(new_helpful)
    df['helpful'] = new_helpful_list
    reviews = df[df['helpful'] >= 0.4]
    reviews = reviews.drop(['helpful'], axis=1)
    reviews.reset_index(drop=True, inplace=True)
    return reviews


def text_preprocessing(dataset):
    dataset['review_text'] = dataset['review_text'].map(
        lambda x: re.sub('[^a-zA-Z]', ' ', str(x)))
    dataset['review_text'] = dataset['review_text'].str.split().map(
        lambda sl: " ".join(s for s in sl if len(s) > 3))
    dataset['review_text'] = dataset['review_text'].map(
        lambda x: re.sub('['+string.punctuation+']', ' ', str(x)))
    dataset['review_text'] = dataset['review_text'].map(
        lambda x: str(x).lower())
    dataset['review_text'] = dataset['review_text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    tokens = [word_tokenize(word) for word in dataset.review_text]
    lemmatizer = WordNetLemmatizer()
    dataset['review_text'] = [
        [lemmatizer.lemmatize(token) for token in t] for t in tokens]

    return dataset


def build_model(inputs, embedding_dim, sequence_length):
    model = models.Sequential([
        layers.Embedding(input_dim=inputs,
                         output_dim=embedding_dim, input_length=sequence_length, mask_zero=True),
        layers.Bidirectional(layer=layers.LSTM(embedding_dim, dropout=0.5)),
        layers.Dense(embedding_dim, activation='relu'),
        layers.Dense(1)])

    return model


def encode_rating(ds):
    ds['rating'].replace(
        {1: 'negative', 2: 'negative', 4: 'positive', 5: 'positive'}, inplace=True)
    ds['rating'].replace({"negative": 0, "positive": 1}, inplace=True)


def encode_review(x, tokenizer):
    x_seq = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x_seq, maxlen=2000, padding='post')

    return x


def set_hyper_params(x_train):
    train_words = [word for tokens in x_train
                   for word in tokens]
    word_length = [len(tokens) for tokens in x_train]
    TRAINING_VOCAB = sorted(list(set(train_words)))
    vocab = len(TRAINING_VOCAB)
    max_len = max(word_length)

    return (max_len, vocab)


def train(ds):

    encode_rating(ds)

    x = ds['review_text']
    y = ds['rating']

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)
    max_len, vocab = set_hyper_params(x_train)

    tokenizer = Tokenizer(num_words=vocab)
    tokenizer.fit_on_texts(x)

    train_word_index = tokenizer.word_index

    x_train = encode_review(x_train, tokenizer)
    x_val = encode_review(x_val, tokenizer)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = build_model(len(train_word_index), 64, 2000)
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=32,
              epochs=2, shuffle=True)

    model.save('model.h5')

    loss, accuracy = model.evaluate(x_val, y_val)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


dataset = process_df(df)
final_dataset = text_preprocessing(dataset)
train(final_dataset)
