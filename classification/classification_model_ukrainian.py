import re

import gensim
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
)
from sklearn.utils import shuffle
import spacy

import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

pandas.set_option('display.max_colwidth', 60)


def load_twitter_messages() -> pandas.DataFrame:
    return shuffle(pandas.read_csv('twitter_dataset_translated.csv'))


nlp = spacy.load('xx_ent_wiki_sm', disable=['ner', 'parser'])


def preprocess_message_advanced(x: str) -> list[str]:
    if not x or not isinstance(x, str):
        return []

    document = nlp(x)

    clean_tokens = [
        token.text.lower()
        for token in document
        if not token.is_stop and not token.is_punct
    ]

    return clean_tokens


def make_agg_vectors_from_text(
    word2vector_model: gensim.models.Word2Vec,
    text_data: pandas.Series,
) -> numpy.ndarray:
    existing_words_in_model = set(word2vector_model.wv.index_to_key)

    return numpy.array([
        numpy.array([
            word2vector_model.wv[word]
            for word in words
            if word in existing_words_in_model
        ])
        for words in text_data
    ])


def make_mean_vector(v: numpy.ndarray) -> numpy.ndarray:
    if v.size:
        return v.mean(axis=0)
    else:
        return numpy.zeros(100, dtype=float)


def make_mean_vectors(vectors: numpy.ndarray) -> list[numpy.ndarray]:
    return list(map(make_mean_vector, vectors))


messages = load_twitter_messages()

messages['text_processed'] = messages['text'].apply(preprocess_message_advanced)
messages['text_processed'].dropna()


X_train, X_test, y_train, y_test = train_test_split(
    messages['text_processed'],
    messages['is_propaganda'],
    test_size=0.2,
)


word2vector_model = gensim.models.Word2Vec(
    messages['text_processed'],
    vector_size=100,
    window=5,
    min_count=2
)

word2vector_model.save('w2v_model_propaganda_twitter')


X_train_vectors = make_mean_vectors(make_agg_vectors_from_text(word2vector_model, X_train))
X_test_vectors = make_mean_vectors(make_agg_vectors_from_text(word2vector_model, X_test))


random_forest = RandomForestClassifier()
random_forest.fit(X_train_vectors, y_train.values.ravel())


y_pred = random_forest.predict(X_test_vectors)


def accuracy_score(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> float:
    return (y_pred == y_true).sum() / len(y_pred)


precision = round(precision_score(y_test, y_pred), 3)
recall = round(recall_score(y_test, y_pred), 3)
accuracy = round(accuracy_score(y_test, y_pred), 3)

print(
    f'Precision: {precision}\n'
    f'Recall: {recall}\n '
    f'Accuracy: {accuracy}'
)


rfc_display = PrecisionRecallDisplay.from_estimator(
    random_forest,
    X_test_vectors,
    y_test.values.ravel(),
    name="Класифікатор RandomForest"
)
_ = rfc_display.ax_.set_title("Крива метрик Precision-Recall")
handles, labels = rfc_display.ax_.get_legend_handles_labels()

plt.show()
