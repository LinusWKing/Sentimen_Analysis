import os
import pickle
import re
import string
from this import d
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
from spellchecker import SpellChecker

nltk.download('stopwords')
nltk.download('wordnet')
stop = stopwords.words('english')


# Process our input to allow for Xml parsing
def process_text(text):
    for punctuation in '&;':
        # Remove the & and ; to avoid parsing errors
        text = text.replace(punctuation, '')
    printable = set(string.printable)
    # Remove non-Ascii characters
    s = list(filter(lambda x: x in printable, text))
    processed_text = ''.join([str(e) for e in s])
    # Add a tag to help structure the Xml Content Better
    processed_text = '<data>\n' + processed_text + '</data>'
    return processed_text

# Find our review files


def load_data():
    # traverse root directory, and list directories as dirs and files as files
    for root, dir, files in os.walk("."):
        for file in files:
            if file.endswith('.review') and not file.startswith('unlabeled'):
                df = parse_xml(os.path.join(root, file))

    return pd.concat(df)


tmp_df = []

# Convert file content to a dataframe


def parse_xml(path):
    with open(path, 'r') as file:
        text = file.read()
        text = process_text(text)
        df = pd.read_xml(text)
        tmp_df.append(df)
    file.close()

    return tmp_df


df = load_data()


# Process our dataframe
def process_df(df):

    cols = ['helpful', 'rating', 'review_text']  # List of columns to use
    dropped_columns = [col for col in df.columns if col not in cols]

    for col in dropped_columns:
        df.drop([col], axis=1, inplace=True)  # Drop columns not used

    df.dropna(inplace=True)  # Remove null values
    df.drop_duplicates(inplace=True)  # Remove duplicates

    # Remove reviews with a low helpful rating
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
    reviews = reviews.drop(['helpful'], axis=1)  # Drop the helpful column
    reviews.reset_index(drop=True, inplace=True)
    return reviews


# Process the text data
def text_preprocessing(dataset):
    dataset['review_text'] = dataset['review_text'].map(
        lambda x: re.sub('[^a-zA-Z]', ' ', str(x)))  # Remove numbers
    dataset['review_text'] = dataset['review_text'].str.split().map(
        lambda sl: " ".join(s for s in sl if len(s) > 3))  # Remove words with less than 3 characters
    dataset['review_text'] = dataset['review_text'].map(
        lambda x: re.sub('['+string.punctuation+']', ' ', str(x)))  # Remove punctuation
    dataset['review_text'] = dataset['review_text'].map(
        lambda x: str(x).lower())  # Convert to lower-case
    dataset['review_text'] = dataset['review_text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))  # Remove stop-words
    tokens = [word_tokenize(word)
              for word in dataset.review_text]  # Tokenize the data

    # Check and correct spelling errors. Consider commenting out as it takes a while.

    spell = SpellChecker()
    for i in range(len(tokens)):
        for j, token in enumerate(tokens[i]):
            if token not in spell:
                tokens[i][j] = spell.correction(token)

    # Lemmatize our tokens
    lemmatizer = WordNetLemmatizer()
    dataset['review_text'] = [
        [lemmatizer.lemmatize(token) for token in t] for t in tokens]

    return dataset

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

# Encode our rating


def encode_rating(ds):
    ds['rating'].replace(
        {1: 'negative', 2: 'negative', 4: 'positive', 5: 'positive'}, inplace=True)
    ds['rating'].replace({"negative": 0, "positive": 1}, inplace=True)

# Encode our reviews


def encode_review(x, tokenizer):
    x_seq = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x_seq, maxlen=1000, padding='post')

    return x


def train(ds):

    encode_rating(ds)

    x = ds['review_text']
    y = ds['rating']

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)

    train_word_index = tokenizer.word_index

    x_train = encode_review(x_train, tokenizer)
    x_val = encode_review(x_val, tokenizer)

    # Save our toeknizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = build_model(len(train_word_index)+1, 64, 1000)

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=32,
              epochs=2, shuffle=True)

    model.save('model.h5')  # Save the model

    loss, accuracy = model.evaluate(x_val, y_val)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


dataset = process_df(df)
final_dataset = text_preprocessing(dataset)
train(final_dataset)
