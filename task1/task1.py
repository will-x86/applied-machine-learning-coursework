# pyright: basic
import os
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from lib import utils

os.environ["SSL_CERT_FILE"] = (
    "/etc/ssl/certs/ca-bundle.crt"  # Will almost defo break your machine
)

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

p_v = "./sentiment_analysis_validation_data.csv"
p_test = "./sentiment_analysis_test_data.csv"
p_t = "./sentiment_analysis_training_data.csv"

STOP = set(stopwords.words("english"))
LEMMA = WordNetLemmatizer()


def get_confusion_matrix(true_label, pred_label):
    return confusion_matrix(true_label, pred_label)


def preprocess(text):
    text = text.lower()  # .. lower
    text = re.sub(r"\d+", "NUM", text)  # Replace #'s with NUM
    tokens = re.findall(r"\b[a-z]+\b", text)  # Extract words
    tokens = [LEMMA.lemmatize(t) for t in tokens if t not in STOP]
    return " ".join(tokens)


def run_task1():
    text_train, labels_train, text_val, labels_val, text_test = utils.load_data_train(
        path_val=p_v, path_test=p_test, path_train=p_t
    )

    labels_train = np.array(labels_train)
    spam_mask = np.array([detect_spam([t])[0] for t in text_train])
    clean_texts = [t for t, s in zip(text_train, spam_mask) if s == 0]
    clean_labels = labels_train[spam_mask == 0]
    reduced_words_count = count_vocabulary_reduction(clean_texts)
    print(f"lemming & preprocessing removed {reduced_words_count} unique words")

    clean_texts = [preprocess(t) for t in clean_texts]
    val_texts = [preprocess(t) for t in text_val]

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_train = tfidf.fit_transform(clean_texts)
    vizualize_tfidf(tfidf, X_train)
    X_val = tfidf.transform(val_texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, clean_labels)

    val_spam = detect_spam(text_val)
    val_preds = np.where(val_spam == -1, -1, clf.predict(X_val))

    val_mask = val_preds != -1
    print(get_confusion_matrix(np.array(labels_val)[val_mask], val_preds[val_mask]))


reg = r"(?i)^(subjects?|from|forwarded by)\s*"


def detect_spam(train):
    compiled = re.compile(reg)
    result = []
    for x, y in enumerate(train):
        val = compiled.search(y)
        result.append(-1 if val is not None else 0)
    return result


def count_vocabulary_reduction(raw_texts):
    words_pre = set()
    words_post = set()

    for text in raw_texts:
        words_pre.update(re.findall(r"\b[a-z]+\b", text.lower()))

        processed_text = preprocess(text)
        words_post.update(processed_text.split())

    return len(words_pre) - len(words_post)


def vizualize_tfidf(tfidf, X_train):
    feature_names = np.array(tfidf.get_feature_names_out())
    mean_tfidf = np.asarray(X_train.mean(axis=0)).flatten()
    top_idx = mean_tfidf.argsort()[-20:]

    _, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_names[top_idx], mean_tfidf[top_idx])
    ax.set_xlabel("Mean TF-IDF score")
    ax.set_title("Top 20 TF-IDF terms")
    plt.tight_layout()
    plt.savefig("tfidf.png", dpi=150)
