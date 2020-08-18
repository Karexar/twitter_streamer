import os
import pandas as pd
from corpus_class.corpus_stat import *

class Corpus_8:
    """This class handles the GSW, german, and 6 other GSW-like corpus"""
    def __init__(self, dir_path):
        corpus_names = ["gsw", "deu", "bar", "frr", "ksh", "lim", "nds", "pfl"]
        corpus_names = [x for x in os.listdir(dir_path) if x[:3] in corpus_names
                        and x.endswith(".txt")]
        if len(corpus_names) != 8:
            raise Exception("Some files cannot be found")
        self.corpus = dict()
        print("Loading corpus...")
        for name in corpus_names:
            print(name)
            path = os.path.join(dir_path, name)
            self.corpus[name[:3]] = Corpus_stat(path, sep="\t")
        self.languages = self.corpus.keys()

    def get_gsw_specific_words(self,
                               specificity_th,
                               count,
                               path_speakers=None):
        """Use a proportion strategy to compute the most frequent words in GSW
        that are not frequent in German or other GSW-like languages.
        """

        print("Computing common GSW words...")
        gsw_words = self.corpus["gsw"].get_common_words(10000)

        # Create the dataframe containing proportion of words for each language
        df = pd.DataFrame(index=gsw_words)
        print("Computing the proportions of words for each language")
        for language in self.languages:
            print(language)
            df[language] = df.index.map(self.corpus[language].get_proportion)

        # Load the speaker counts to weight the proportions
        if path_speakers is not None:
            df_speakers = pd.read_csv(path_speakers, index_col="language_code")
            for language in self.languages:
                df[language] = df[language] * df_speakers["speakers"][language]

        df["sum_ratio"] = df[self.languages].sum(axis=1)
        df["gsw_ratio"] = df["gsw"] / df["sum_ratio"]

        filtered = df[df["gsw_ratio"] > specificity_th]
        final = filtered["gsw"].sort_values(ascending=False)
        common_words = final.index[:400]
        if len(common_words) < 400:
            raise Exception("Not enough common words, you may need to increase " +
                  "the amount of GSW words used in the dataframe")
        return list(common_words)
