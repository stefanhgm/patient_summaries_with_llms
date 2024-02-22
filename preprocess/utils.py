import string
import spacy
import nltk
from nltk.util import ngrams
import pandas as pd
import numpy as np
from spacy.symbols import ORTH
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import binarize
from tqdm import tqdm
from src.preprocess.regular_expressions import *
    

def get_pairwise_text_similarity(texts, batch_size=1000, threshold=0):
    """ Calculates the pairwise cosine similarity between the texts. Use batch sizes and threshold to reduce memory usage."""
    vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='unicode', stop_words='english') # , max_features=1000)

    X = vectorizer.fit_transform(texts).astype(np.float32)
    S = lil_matrix((len(texts), len(texts)), dtype=np.float32)
    # Determine cosine_similarity in batches and remove unnecessary values via thresholding
    pbar = tqdm(total=len(range(0, len(texts), batch_size)))
    for i in range(0, len(texts), batch_size):
        S[:, i:i+batch_size] = binarize(cosine_similarity(X, X[i:i+batch_size], dense_output=False), threshold=threshold)
        pbar.update(1)

    return S.tocsr()


def get_most_frequent_words(texts, limit=None):
    """ Use spacy to count the n most frequent words in the texts. """
    if limit is not None:
        texts = texts.sample(limit, replace=True)
    nlp = spacy.load('en_core_web_sm')
    docs = [doc for doc in nlp.pipe(texts.tolist(), n_process=4)]
    counts = [doc.count_by(ORTH).items() for doc in docs]
    words = [word for count in counts for word, _ in count]
    counts = [count for count in counts for _, count in count]
    df_counts = pd.DataFrame({'word': words, 'count': counts})
    # Group by words and keep word
    df_counts = df_counts.groupby('word', as_index=False).sum().sort_values('count', ascending=False)
    df_counts['word'] = df_counts['word'].apply(lambda x: nlp.vocab.strings[x])
    # Sort
    df_counts = df_counts.sort_values('count', ascending=False, ignore_index=True)
    return df_counts


def get_overlapping_ngram_spans(texts, n_gram_length=20, n_gram_min_occurence=20):
    """ Detects spans of ngrams that occur frequently in the texts."""
    tokenize = lambda x: nltk.word_tokenize(x)

    tokens = tokenize((' ' + 10*'#' + ' ').join(texts))
    n_grams = ngrams(tokens, n_gram_length)
    fdist = nltk.FreqDist(n_grams)
    frequent_n_grams = set(ngram for ngram, count in fdist.items() if count > n_gram_min_occurence)

    # Detect spans of ngrams in texts
    duplicate_spans = Counter()
    for summ_tokens in [list(tokenize(summ)) for summ in texts]:
        last_start_duplicate = -1
        for i in range(0, len(summ_tokens) - n_gram_length):
            ngram = tuple(summ_tokens[i:i+n_gram_length])
            if ngram in frequent_n_grams:
                if last_start_duplicate == -1:
                    # Begin of new duplicate span
                    last_start_duplicate = i
            if not (ngram in frequent_n_grams) or i == len(summ_tokens) - n_gram_length - 1:
                # End of duplicate span
                if last_start_duplicate > -1:
                    duplicate_spans[tuple(summ_tokens[last_start_duplicate:i+n_gram_length-1])] += 1
                    last_start_duplicate = -1
    return duplicate_spans


def split_into_paragraphs(summary):
    # If paragraph smaller than minimum length combin with previous paragraph
    min_num_words = 12
    anonymization = ['___']
    punctuation = list(string.punctuation)
    # TODO: Could add stemming
    # TODO: Could add stopwords
    # stopwords = nltk.corpus.stopwords.words('english')
    ignore_words = punctuation + anonymization
    tokenize = lambda x: [res for res in [t.lower() for t in nltk.word_tokenize(x)] if res not in ignore_words]
    paragraphs = re_paragraph.split(summary)
    paragraphs.reverse()
    for i in range(len(paragraphs)-1, 0, -1):
        if len(tokenize(paragraphs[i])) < min_num_words:
            paragraphs[i-1] = paragraphs[i] + '\n\n' + paragraphs[i-1]
            paragraphs = paragraphs[:i] + paragraphs[i+1:]
    # Check first paragraph
    if len(paragraphs) > 1 and len(tokenize(paragraphs[0])) < min_num_words:
        paragraphs[1] = paragraphs[1] + '\n\n' + paragraphs[0]
        paragraphs = paragraphs[1:]
    paragraphs.reverse()
    return paragraphs
