import pandas as pd
from typing import List
from preprocessing.cleaner import *
from tqdm import tqdm

class Corpus_stat:
    """Tool to compute statistics on a text corpus
    """

    def __init__(self, path: str, sep=","):
        """Load the corpus and initialize attributes.
        Parameters
            path - str
                Path of the csv file containing the corpus. Must contains
                at least a column 'text'
        """
        self.df = pd.read_csv(path, sep=sep)
        self.word_distrib = None
        self.vocab = None
        self.sentences_set = None

    def _sentence_to_word_set(self):
        """Convert the list of sentences into a list of set containing the
        words for each sentences"""
        self.sentences_set = []
        for sentence in self.df.iloc[:, 0].values:
            sentence = Cleaner._isolate_words(sentence)
            words_in_sentence = set(sentence.split())
            self.sentences_set.append(words_in_sentence)

    def _corpus_to_string(self):
        """Concatenate the sentences of the corpus and clean the punctuation and
        digits"""
        full_str = ' '.join(list(self.df.iloc[:, 0].values))
        return Cleaner._isolate_words(full_str)

    def _compute_word_distribution(self):
        """Split the corpus into words and compute the count of each
        word in the corpus. The first call will take longer because
        the word distribution needs to be computed.
        """
        distrib = dict()
        full_str = self._corpus_to_string()
        self.word_distrib = pd.Series(full_str.split()) \
                              .value_counts() \
                              .sort_values(ascending=False)
        if self.vocab is None:
            self.vocab = set(self.word_distrib.index)

    def get_common_words(self, count=None):
        """Return the list of words that appear the most often in the
        corpus"""
        if self.word_distrib is None:
            self._compute_word_distribution()
        if count is None:
            return list(self.word_distrib.index)
        return list(self.word_distrib[:count].index)

    def get_vocabulary(self):
        """Return the set of words that appear in the corpus"""
        if not self.vocab:
            full_str = self._corpus_to_string()
            self.vocab = set(full_str.split())
        return self.vocab

    def get_coverage(self, words):
        """Compute the coverage, which is the proportion of sentences in the
        corpus that contain at least one word from the words list."""
        if self.sentences_set is None:
            self._sentence_to_word_set()
        words = set(words)
        match = 0
        for word_set in tqdm(self.sentences_set):
            if len(word_set.intersection(words)) > 0:
                match += 1
        return match / len(self.sentences_set)

    def get_proportion(self, word):
        """Get the proportion of a word in the corpus"""
        if self.word_distrib is None:
            self._compute_word_distribution()
        if word in self.word_distrib:
            return self.word_distrib[word] / self.word_distrib.sum()
        return 0.0
