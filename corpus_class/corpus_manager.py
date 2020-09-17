import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from utils.utils import *
from preprocessing.cleaner import *
from corpus_class.corpus_stat import *
from tqdm import tqdm


class CorpusManager:
    """A class containing corpus for different languages, allowing to compute
    the most specific words for a given language, using a TF-IDF or proportion-
    based approach.
    """

    def __init__(self, dir_path, path_speakers, langs=None):
        """
        Parameters
            dir_path - str
                The directory containing the corpus file. Each file has one
                sentence per line.
            path_speakers - str
                A path to a csv file containing the language_code in the first
                column, and the people count speaking that language on the
                second column. This is used to weight the scores for each word
                of each language. If None, no ponderation is applied.
            langs - List[str]
                A list of language codes (e.g. "gsw") that will be used for
                computing the most common words. The code corresponds to the
                three first letter of the file names in 'dir_path'
        """

        self.ponderation = dict()
        self.lang_to_corpus = dict()
        self.path_speakers = path_speakers
        if langs is None:
            langs = [x[:3] for x in os.listdir(dir_path) if x.endswith("txt")]
        for doc in os.listdir(dir_path):
            if doc.endswith(".txt") and doc[:3] in langs:
                path = os.path.join(dir_path, doc)
                print("Creating corpus stat for " + path)
                self.lang_to_corpus[doc[:3]] = Corpus_stat(path, sep="\t")
        self.langs = self.lang_to_corpus.keys()
        self.df_tfidf = None
        self.df_prop = None


    def _compute_tfidf_scores(self):
        docs = [self.lang_to_corpus[lang].full_str for lang in self.langs]
        print("Vectorization...")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names()

        self.df_tfidf = pd.DataFrame(X.T.todense(),
                                     index=feature_names,
                                     columns=self.langs)
        self.df_tfidf = self.df_tfidf[self.df_tfidf.index.map(len) > 2]

        # Weight the scores by the amount of people speaking the language
        if not self.path_speakers is None:
            df_speakers = pd.read_csv(self.path_speakers, index_col="language_code")
            for lang in self.langs:
                ponderated = self.df_tfidf[lang] * df_speakers["speakers"][lang]
                self.df_tfidf[lang] = ponderated

    def get_specific_words_tfidf(self,
                                 language,
                                 threshold,
                                 count):
        """Compute the most common words that are specific to a given language
        using the tfidf method.

        Parameters
            language - str
                The language code (e.g. "gsw") for which we want to compute the
                most common words.
            threshold - float
                A value between 0 and 1 to filter out words that are not
                specific enough to the given language.
            count - int
                How many words we want to get

        Returns
            List[str]
                The list of most common words specific to the given language.
        """
        if self.df_tfidf is None:
            self._compute_tfidf_scores()

        if not language + "_ratio" in self.df_tfidf.columns:
            scores = self.df_tfidf[language]
            sum_scores = self.df_tfidf[self.langs].sum(axis=1)
            self.df_tfidf[language + "_ratio"] = scores / sum_scores

        # Filter out the words with not enough language specificity
        mask = self.df_tfidf[language + "_ratio"] > threshold
        df_filtered = self.df_tfidf[mask]

        # Keep the words with the highest TF-IDF score
        words = df_filtered[language].sort_values(ascending=False)[:count]

        return list(words.index)

    def get_specific_words_prop(self,
                                language,
                                threshold,
                                count):
        """Compute the most common words that are specific to a given language
        using the proportion method.

        Parameters
            language - str
                The language code (e.g. "gsw") for which we want to compute the
                most common words.
            threshold - float
                A value between 0 and 1 to filter out words that are not
                specific enough to the given language.
            count - int
                How many words we want to get

        Returns
            List[str]
                The list of most common words specific to the given language.
        """

        print(f"Computing common words for {language}...")
        # Get the most common words for this language, we don't compute on all
        # words for efficiency
        lang_words = self.lang_to_corpus[language].get_common_words(10000)

        # Create the dataframe containing proportion of words for each language
        if self.df_prop is None:
            self.df_prop = pd.DataFrame(index=lang_words)
            print("Computing the proportions of words for each language")
            for lang in self.langs:
                print(lang)
                f = self.lang_to_corpus[lang].get_proportion
                self.df_prop[lang] = self.df_prop.index.map(f)

            # Load the speaker counts to weight the proportions
            if self.path_speakers is not None:
                df_speakers = pd.read_csv(self.path_speakers,
                                          index_col="language_code")
                for lang in self.langs:
                    self.df_prop[lang] = (self.df_prop[lang] *
                                              df_speakers["speakers"][lang])

            self.df_prop["sum_scores"] = self.df_prop[self.langs].sum(axis=1)

        if not language + "_ratio" in self.df_prop:
            ratios = self.df_prop[language] / self.df_prop["sum_scores"]
            self.df_prop[language+"_ratio"] = ratios
        filtered = self.df_prop[self.df_prop[language + "_ratio"] > threshold]
        final = filtered["gsw"].sort_values(ascending=False)
        common_words = final.index[:400]
        if len(common_words) < 400:
            raise Exception("Not enough common words, you may need to " +
                  "increase the amount of most common words used in the " +
                  "dataframe")
        return list(common_words)
