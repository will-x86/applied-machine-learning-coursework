# pyright: basic
import os
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
from huggingface_hub import login
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import LinearSVC

from lib import utils

# tfidf_max_features = 50000
tfidf_max_features = 50000
regex_contamination_percent = 0.259
seed = 42

if (host := os.getenv("HOST")) is not None and "framework" in host:
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-bundle.crt"
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = (
        "11.0.0"  # rdna 2 laptop - shucks - disable if on rdna3 (90XX series AMD)
    )

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

p_v = "./sentiment_analysis_validation_data.csv"
p_test = "./sentiment_analysis_test_data.csv"
p_t = "./sentiment_analysis_training_data.csv"

STOP = set(stopwords.words("english"))
# STOP = set(stopwords.words("english")) - {"no", "not", "never", "nor", "neither", "without"} - made it worse .. ?

LEMMA = WordNetLemmatizer()
reg = r"(?i)^(subjects?|from|forwarded by)\s*"


def get_confusion_matrix(true_label, pred_label):
    return confusion_matrix(true_label, pred_label)


def preprocess(text):  # Ironically barely 1% difference
    text = text.lower()
    text = re.sub(r"\d+", "NUM", text)
    tokens = re.findall(r"\b[a-z]+\b", text)
    tokens = [LEMMA.lemmatize(t) for t in tokens if t not in STOP]
    return " ".join(tokens)


def remove_spam_regex(texts):
    compiled = re.compile(reg)
    return np.array([-1 if compiled.search(t) else 0 for t in texts])


def remove_spam_isolation_tfidf(train_texts, target_texts=None):
    if target_texts is None:
        target_texts = train_texts
    vec = TfidfVectorizer(max_features=tfidf_max_features, sublinear_tf=True)
    X_tr = vec.fit_transform(train_texts)
    X_tg = vec.transform(target_texts)
    # pyright will be the death of me
    forest = IsolationForest(
        contamination=regex_contamination_percent, random_state=seed  # pyright: ignore
    )
    preds = forest.fit(X_tr).predict(X_tg)
    return np.where(preds == 1, 0, -1)


def train_and_eval_sbert(
    clean_texts, clean_labels, val_texts, val_labels, val_spam, name
):
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = sbert.encode(list(clean_texts), show_progress_bar=True)
    X_val = sbert.encode(list(val_texts), show_progress_bar=True)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, clean_labels)
    preds = np.where(val_spam == -1, -1, clf.predict(X_val))
    val_mask = preds != -1
    cm = get_confusion_matrix(np.array(val_labels)[val_mask], preds[val_mask])
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    print(
        f"\n[{name}-sbert] spam removed: {(val_spam==-1).sum()} | accuracy: {acc:.3f}"
    )
    print(cm)
    return cm


def train_and_eval_svm(
    clean_texts,
    clean_labels,
    val_texts,
    val_labels,
    val_spam,
    name,
):
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), max_features=tfidf_max_features, sublinear_tf=True
    )
    X_train = tfidf.fit_transform([preprocess(t) for t in clean_texts])
    vizualize_tfidf(tfidf, X_train, name)
    X_val = tfidf.transform([preprocess(t) for t in val_texts])

    clf = LinearSVC(random_state=seed, max_iter=2000)
    clf.fit(X_train, clean_labels)
    preds = np.where(val_spam == -1, -1, clf.predict(X_val))
    val_mask = preds != -1
    cm = get_confusion_matrix(np.array(val_labels)[val_mask], preds[val_mask])
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    print(f"\n[{name}-svm] spam removed: {(val_spam==-1).sum()} | accuracy: {acc:.3f}")
    print(cm)
    return cm


def train_and_eval_gemma(
    clean_texts, clean_labels, val_texts, val_labels, val_spam, name
):
    login()
    sbert = SentenceTransformer("google/embeddinggemma-300M")
    X_train = sbert.encode(
        list(clean_texts), prompt_name="Classification", show_progress_bar=True
    )
    X_val = sbert.encode(
        list(val_texts), prompt_name="Classification", show_progress_bar=True
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, clean_labels)
    preds = np.where(val_spam == -1, -1, clf.predict(X_val))
    val_mask = preds != -1
    cm = get_confusion_matrix(np.array(val_labels)[val_mask], preds[val_mask])
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    print(
        f"\n[{name}-gemma] spam removed: {(val_spam==-1).sum()} | accuracy: {acc:.3f}"
    )
    print(cm)
    return cm


def train_and_eval_logistic(
    clean_texts,
    clean_labels,
    val_texts,
    val_labels,
    val_spam,
    name,
):
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), max_features=tfidf_max_features, sublinear_tf=True
    )
    X_train = tfidf.fit_transform([preprocess(t) for t in clean_texts])
    vizualize_tfidf(tfidf, X_train, name)
    X_val = tfidf.transform([preprocess(t) for t in val_texts])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, clean_labels)
    preds = np.where(val_spam == -1, -1, clf.predict(X_val))
    val_mask = preds != -1
    cm = get_confusion_matrix(np.array(val_labels)[val_mask], preds[val_mask])
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    print(
        f"\n[{name}-logistic] spam removed: {(val_spam==-1).sum()} | accuracy: {acc:.3f}"
    )
    print(cm)
    return cm


def count_vocabulary_reduction(raw_texts):
    words_pre, words_post = set(), set()
    for text in raw_texts:
        words_pre.update(re.findall(r"\b[a-z]+\b", text.lower()))
        words_post.update(preprocess(text).split())
    return len(words_pre) - len(words_post)


def visualize_isolation_forest(train_texts, name="IsolationForest"):
    vec = TfidfVectorizer(max_features=tfidf_max_features, sublinear_tf=True)
    X = vec.fit_transform(train_texts)

    preds = IsolationForest(
        contamination=regex_contamination_percent, random_state=seed  # pyright: ignore
    ).fit_predict(X)
    labels = np.where(preds == 1, "review", "spam")

    X_2d = PCA(n_components=2, random_state=seed).fit_transform(
        X.toarray()  # pyright: ignore
    )

    _, ax = plt.subplots(figsize=(8, 6))
    for label, color in [("review", "steelblue"), ("spam", "tomato")]:
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=label, alpha=0.4, s=10)

    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title(f"Isolation Forest spam detection - {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{name}-isolation_forest.png", dpi=150)


def vizualize_tfidf(tfidf, X_train, name):
    feature_names = np.array(tfidf.get_feature_names_out())
    mean_tfidf = np.asarray(X_train.mean(axis=0)).flatten()
    top_idx = mean_tfidf.argsort()[-20:]
    _, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_names[top_idx], mean_tfidf[top_idx])
    ax.set_xlabel("Mean TF-IDF score")
    ax.set_title(f"Top 20 TF-IDF terms - {name}")
    plt.tight_layout()
    plt.savefig(f"{name}-tfidf.png", dpi=150)


def run_task1():
    text_train, labels_train, text_val, labels_val, _text_test = utils.load_data_train(
        path_val=p_v, path_test=p_test, path_train=p_t
    )

    print(
        f"vocab reduction: {count_vocabulary_reduction(text_train)} unique words removed"
    )
    # regex to remove spam
    mask_regex = remove_spam_regex(text_train)
    print(
        f"Regex flagged {sum(1 for s in mask_regex if s == -1)} of {len(text_train)} train samples as spam "
    )
    #    print("Non-Spam Samples (regex)")
    #    for text, mask_val in zip(text_train, mask_regex):
    #        if mask_val == 0:
    #            print(f"- {text}")

    clean_texts_r = [t for t, s in zip(text_train, mask_regex) if s == 0]
    clean_labels_r = np.array(labels_train)[mask_regex == 0]
    val_spam_regex = remove_spam_regex(text_val)
    cm_regex_logistic = train_and_eval_logistic(
        clean_texts_r, clean_labels_r, text_val, labels_val, val_spam_regex, "Regex"
    )

    # Isolation Forest to remove spam
    mask_iso = remove_spam_isolation_tfidf(text_train)
    visualize_isolation_forest(text_train)

    clean_texts_i = [t for t, s in zip(text_train, mask_iso) if s == 0]
    clean_labels_i = np.array(labels_train)[mask_iso == 0]
    val_spam_iso = remove_spam_isolation_tfidf(text_train, text_val)
    cm_iso_logistic = train_and_eval_logistic(
        clean_texts_i,
        clean_labels_i,
        text_val,
        labels_val,
        val_spam_iso,
        "IsolationForest",
    )
    cm_regex_svm = train_and_eval_svm(
        clean_texts_r, clean_labels_r, text_val, labels_val, val_spam_regex, "Regex"
    )
    cm_iso_svm = train_and_eval_svm(
        clean_texts_i,
        clean_labels_i,
        text_val,
        labels_val,
        val_spam_iso,
        "IsolationForest",
    )

    cm_regex_bertyboi = train_and_eval_sbert(
        clean_texts_r, clean_labels_r, text_val, labels_val, val_spam_regex, "Regex"
    )
    cm_iso_bertyboi = train_and_eval_sbert(
        clean_texts_i,
        clean_labels_i,
        text_val,
        labels_val,
        val_spam_iso,
        "IsolationForest",
    )
    cm_regex_gemma = train_and_eval_gemma(
        clean_texts_r, clean_labels_r, text_val, labels_val, val_spam_regex, "Regex"
    )
    _, axes = plt.subplots(2, 4, figsize=(12, 10))
    cms = [
        cm_regex_logistic,
        cm_regex_svm,
        cm_regex_bertyboi,
        cm_iso_logistic,
        cm_iso_svm,
        cm_iso_bertyboi,
        cm_regex_gemma,
    ]
    titles = [
        "Regex + Logistic Regression",
        "Regex + SVM",
        "Regex + Berty",
        "IsolationForest + Logistic Regression",
        "IsolationForest + SVM",
        "IsolationForest + Berty",
        "Regex + Gemma",
    ]

    for ax, cm, title in zip(axes.ravel(), cms, titles):
        ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"]).plot(
            ax=ax, colorbar=False
        )
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("comparison_confusion.png", dpi=150)
