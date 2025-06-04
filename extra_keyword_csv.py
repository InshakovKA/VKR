import os
import numpy as np
import pandas as pd
import nltk
from nltk.tag import pos_tag, pos_tag_sents
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from nltk.tokenize import RegexpTokenizer

def average_sentence(lst):
    length = 0
    count = 0
    for sent in lst:
        count += 1
        length += len(nltk.word_tokenize(sent, language="english"))
    return length / count


def average_word_len(lst):
    length = 0
    count = 1
    for word in lst:
        count += 1
        length += len(word)
    return length / count


def main():
    #nltk.download('averaged_perceptron_tagger')
    X = []
    morph = nltk.stem.WordNetLemmatizer()
    articles = "fake_arts"
    fake_count = 0
    real_count = 0
    most_popular = []
    average_sentences = []
    prop_rate = []
    adj_rate = []
    adverb_rate = []
    average_word = []
    noun_rate = []
    verb_rate = []
    titles = []
    title_count_norms = []
    keywords = ['said', 'he', 'like', 'one', 'new', 'would', 'time', 'but', 'could', 'it']
    key_counts = {}
    for i in range(len(keywords)):
            key_counts[f"word{i}"] = []

    for file in sorted(os.listdir(articles)):
        fake_count += 1
        tokenizer = RegexpTokenizer(r'\w+')
        title_count = CountVectorizer()
        filename = os.path.join(articles, file)
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        words = [wd.lower() for wd in nltk.word_tokenize(text, language="english") if wd.isalpha()]
        sentences = nltk.sent_tokenize(text, language="english")

        first_sent = sentences[0] if (len(sentences[0].split()) > 1 or len(sentences) == 1) else sentences[1]
        titles.append(first_sent.lower())
        title_count.fit(tokenizer.tokenize(first_sent.lower()))
        counts = csr_matrix(title_count.transform([text.lower()]))
        title_count_norms.append(np.linalg.norm(counts.toarray()))

        average_word.append(average_word_len(words))
        average_sentences.append(average_sentence(sentences))

        lemmas = [morph.lemmatize(word) for word in words]
        tagged = pos_tag(nltk.word_tokenize(text))
        prop = [wd for wd, pos in tagged if pos == "NNP" or pos == "NNPS"]
        adj = [wd for wd, pos in tagged if pos == "JJ" or pos == "JJR" or pos == "JJS"]
        adv = [wd for wd, pos in tagged if pos == "RB" or pos == "RBR" or pos == "RBS"]
        nouns = [wd for wd, pos in tagged if pos == "NN" or pos == "NNS"]
        verbs = [wd for wd, pos in tagged if pos in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]

        prop_rate.append(len(prop) / len(words))
        adj_rate.append(len(adj) / len(words))
        adverb_rate.append(len(adv) / len(words))
        noun_rate.append(len(nouns) / len(words))
        verb_rate.append(len(verbs) / len(words))

        for i in range(len(keywords)):
            count = 0
            for lemma in lemmas:
                if lemma == keywords[i]:
                    count += 1
            key_counts[f"word{i}"].append(count)

        unique = list(set(lemmas))
        n_unique = len(unique)
        counter = Counter(lemmas)
        most_popular.append(counter.most_common(1)[0][1] / len(words))
        X.append(text)

    articles = "real_arts"
    for file in sorted(os.listdir(articles)):
        real_count += 1
        tokenizer = RegexpTokenizer(r'\w+')
        title_count = CountVectorizer()
        filename = os.path.join(articles, file)
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
        X.append(text)

        words = [wd.lower() for wd in nltk.word_tokenize(text, language="english") if wd.isalpha()]
        sentences = nltk.sent_tokenize(text, language="english")

        first_sent = sentences[0] if (len(sentences[0].split()) > 1 or len(sentences) == 1) else sentences[1]
        titles.append(first_sent.lower())
        title_count.fit(tokenizer.tokenize(first_sent.lower()))
        counts = csr_matrix(title_count.transform([text.lower()]))
        title_count_norms.append(np.linalg.norm(counts.toarray()))

        average_word.append(average_word_len(words))
        average_sentences.append(average_sentence(sentences))

        lemmas = [morph.lemmatize(word) for word in words]
        tagged = pos_tag(nltk.word_tokenize(text))

        prop = [wd for wd, pos in tagged if pos == "NNP" or pos == "NNPS"]
        adj = [wd for wd, pos in tagged if pos == "JJ" or pos == "JJR" or pos == "JJS"]
        adv = [wd for wd, pos in tagged if pos == "RB" or pos == "RBR" or pos == "RBS"]
        nouns = [wd for wd, pos in tagged if pos == "NN" or pos == "NNS"]
        verbs = [wd for wd, pos in tagged if pos in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]

        prop_rate.append(len(prop) / len(words))
        adj_rate.append(len(adj) / len(words))
        adverb_rate.append(len(adv) / len(words))
        noun_rate.append(len(nouns) / len(words))
        verb_rate.append(len(verbs) / len(words))

        for i in range(len(keywords)):
            count = 0
            for lemma in lemmas:
                if lemma == keywords[i]:
                    count += 1
            key_counts[f"word{i}"].append(count)

        unique = list(set(lemmas))
        n_unique = len(unique)
        counter = Counter(lemmas)
        most_popular.append(counter.most_common(1)[0][1] / len(words))
    X = np.array(X, dtype=object)
    y = np.concatenate((np.ones(fake_count, dtype=int), np.zeros(real_count, dtype=int)))

    feature_dict = {"text": X, "most_common": most_popular, "av_sent_length": average_sentences,
                         "proper_nouns": prop_rate, "adjectives": adj_rate, "adverbs": adverb_rate,
                         "nouns": noun_rate, "verbs": verb_rate, "av_word_length": average_word, "title": titles,
                         "norm": title_count_norms, "y": y}

    feature_dict.update(key_counts)

    data = pd.DataFrame(feature_dict)
    print(data.head())
    data.to_csv("interp_data_extra_keywords.csv")


if __name__ == '__main__':
    main()
