import pickle

import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

# Define our network


def build_model(inputs, embedding_dim, sequence_length):
    model = models.Sequential([
        layers.Embedding(input_dim=inputs,
                         output_dim=embedding_dim, input_length=sequence_length, mask_zero=True),  # Embedding layer with inputs = unique vocabulary
        layers.Bidirectional(layer=layers.LSTM(
            embedding_dim, dropout=0.6, return_sequences=True)),  # RNN layer with LSTM
        layers.Bidirectional(layer=layers.LSTM(embedding_dim, dropout=0.6)),
        # Vanilla layer to help with classification
        layers.Dense(embedding_dim, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1)])  # Output layer

    return model

# Encode our rating as 0 for negative and 1 for positive


def encode_rating(ds):
    ds['rating'].replace(
        {1: 'negative', 2: 'negative', 4: 'positive', 5: 'positive'}, inplace=True)
    ds['rating'].replace({"negative": 0, "positive": 1}, inplace=True)

# Encode our reviews


def encode_review(x, tokenizer):
    x_seq = tokenizer.texts_to_sequences(x)
    # Pad/Truncate where necessary
    x = pad_sequences(x_seq, maxlen=1000, padding='post')

    return x


def train(ds):

    encode_rating(ds)

    x = ds['review_text']
    y = ds['rating']

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.33, random_state=42, stratify=y)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)

    train_word_index = tokenizer.word_index

    x_train = encode_review(x_train, tokenizer)
    x_val = encode_review(x_val, tokenizer)

    # Save our toeknizer
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = build_model(len(train_word_index)+1, 64,
                        1000)  # Instaniate the model

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizers.Adam(), metrics=['accuracy'])  # Configure
    model.summary()

    callbacks = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=1)

    model.fit(x_train, y_train, batch_size=64,
              validation_data=(x_val, y_val), callbacks=callbacks,
              epochs=10, shuffle=True)  # Train

    model.save('models/model.h5')  # Save the model


final_dataset = pd.read_pickle('models/preprocessed.pkl')
train(final_dataset)
