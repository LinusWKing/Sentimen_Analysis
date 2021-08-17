# Contributed by Diana Ladaniak
import re
import string

import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

nltk.download('stopwords')
nltk.download('wordnet')
stop = stopwords.words('english')


df = pd.read_pickle('models/df_data.pkl')  # Load saved dataframe


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
    unknown = [spell.unknown(t) for t in tokens]
    for i, token in enumerate(unknown):
        if token:
            tokens[i] = [spell.correction(token)for token in tokens[i]]

    # Lemmatize our tokens
    lemmatizer = WordNetLemmatizer()
    dataset['review_text'] = [
        [lemmatizer.lemmatize(token) for token in t] for t in tokens]

    return dataset


dataset = process_df(df)
processed = text_preprocessing(dataset)
processed.to_pickle('models/preprocessed.pkl')
