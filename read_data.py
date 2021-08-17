
import os
import string

import pandas as pd

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
            if file.endswith('.review'):
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
df.to_pickle('models/df_data.pkl')  # Save dataframe externally
