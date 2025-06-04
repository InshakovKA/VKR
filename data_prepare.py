import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize


class Lemmatizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, text):
        return [self.wnl.lemmatize(t) for t in word_tokenize(text)]

def main():
    data = pd.read_csv("interp_data_title_keywords.csv")
    data = data.sample(frac=1).reset_index(drop=True)

    texts = data["text"]
    y = data["y"]

    texts_train_nn = texts.iloc[:25000]
    y_train_nn = y.iloc[:25000]

    texts_test_nn = texts.iloc[25000:32000]
    y_test_nn = y.iloc[25000:32000]

    texts_train_tree = texts.iloc[32000:57000]
    y_train_tree = y.iloc[32000:57000]

    texts_test_tree = texts.iloc[57000:]
    y_test_tree = y.iloc[57000:]

    vec = CountVectorizer(tokenizer=Lemmatizer(), strip_accents = 'unicode')
    vec.fit(texts_train_nn.values)
    with open('trained_vectorizer.pkl', 'wb') as file:
        pickle.dump(vec, file)

    data_train_nn = pd.DataFrame({"text": texts_train_nn, "y": y_train_nn})
    data_test_nn = pd.DataFrame({"text": texts_test_nn, "y": y_test_nn})

    data_train_tree = pd.DataFrame({"text": texts_train_tree, "y": y_train_tree})
    data_test_tree = pd.DataFrame({"text": texts_test_tree, "y": y_test_tree})

    data_train_nn.to_csv("texts_train_nn.csv")
    data_test_nn.to_csv("texts_test_nn.csv")

    data_train_tree.to_csv("texts_train_tree.csv")
    data_test_tree.to_csv("texts_test_tree.csv")


if __name__ == '__main__':
    main()