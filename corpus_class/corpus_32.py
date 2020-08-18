import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from utils.utils import *
from preprocessing.cleaner import *


class Corpus_32:
    """A class containing corpus for 32 different languages"""

    def __init__(self, dir_path, path_speakers):
        self.ponderation = dict()
        self.corpus = dict()
        for doc in os.listdir(dir_path):
            if doc.endswith(".txt"):
                with open(os.path.join(dir_path, doc), "r",
                          encoding="utf8") as f:
                    print("Reading " + doc)
                    lines = f.readlines()
                    full_str = Cleaner._isolate_words(' '.join(lines))
                    self.corpus[doc[:3]] = full_str

        print("Vectorization...")
        docs = [self.corpus[x] for x in self.corpus]
        self.corpus_index = [x for x in self.corpus]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names()

        self.df = pd.DataFrame(X.T.todense(), index=feature_names, columns=self.corpus_index)
        self.df = self.df[self.df.index.map(len) > 2]

        self._set_ponderation(path_speakers)

    def _set_ponderation(self, path_speakers):
        # Load the count of speakers for each language
        if path_speakers is not None:
            df_speakers = pd.read_csv(path_speakers, index_col="language_code")
            for lang in self.corpus_index:
                self.df[lang + "_pond"] = self.df[lang] * df_speakers["speakers"][lang]

    def get_language_specific_words(self, language, specificity_th, count, pop_ponderated=False):
        if pop_ponderated:
            lang_spec_pond_str = language + "_pond_specific"
            if lang_spec_pond_str not in self.df.columns:
                corpus_index_pond = [x + "_pond" for x in self.corpus_index]
                self.df[lang_spec_pond_str] = self.df[language + "_pond"] / self.df[corpus_index_pond].sum(axis=1)

            # Filter out the words with not enough language specificity
            df_filtered = self.df[self.df[lang_spec_pond_str] > specificity_th]
        else:
            lang_spec_str = language + "_specific"
            if lang_spec_str not in self.df.columns:
                self.df[lang_spec_str] = self.df[language] / self.df[self.corpus_index].sum(axis=1)

            # Filter out the words with not enough language specificity
            df_filtered = self.df[self.df[lang_spec_str] > specificity_th]

        # Keep the words with the highest TF-IDF score (ponderated or not)
        words = df_filtered[language].sort_values(ascending=False)[:count]

        return list(words.index)
