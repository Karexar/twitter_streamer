# This script computes the most common Swiss-German words that are specific to
# Swiss-German, with respect to German and other GSW-like languages.
# It takes the 10000 most common GSW words and for each language, compute the
# ratio between the weighted proportion of this word and the sum over all
# languages of the word weighted proportion. A specificity threshold is userd
# to filter out words that are not specific enough to GSW. The remaining words
# are sorted by proportion in GSW and a list is created with the 400 most
# common words.

from utils.utils import *
from corpus_class.corpus_stat import *
import pandas as pd
import os
import _thread

###  Settings  ################################################################
gsw_path = "data/leipzig_32/gsw_swisstext_leipzig.txt"
deu_path = "data/leipzig_32/deu-de_web-public_2019_1M_converted.txt"
path_speakers = "data/speakers.csv"
corpus_32_dir = "data/leipzig_32"
#gsw_like = ["bar", "frr", "ksh", "lim", "nds", "pfl", ]
gsw_like = ["afr", "bar", "cat", "dan", "eng", "epo", "est", "fin", "fra", "frr",
         "gle", "glg", "hrv", "isl", "ita", "jav", "knn", "ksh", "lim", "ltz",
         "nds", "nld", "pap", "pfl", "por", "ron", "slv", "spa", "swa", "swe"]
#specificities = [round(x*0.05, 2) for x in range(19)]
#specificities = [round(x*0.005, 2) for x in range(201)]
specificities = [0.25]
weighted = True
###############################################################################

_thread.start_new_thread( keep_alive, tuple() )

print("Loading GSW corpus...")
corpus = dict()
corpus["gsw"] = Corpus_stat(gsw_path, sep="\t")

print("Loading German corpus...")
corpus["deu"] = Corpus_stat(deu_path, sep="\t")
print("Loading GSW-like corpus...")
names = [x for x in os.listdir(corpus_32_dir) if x[:3] in gsw_like
         and x.endswith(".txt")]
if len(names) != len(gsw_like):
    raise Exception("Some files cannot be found")
for name in names:
    path = os.path.join(corpus_32_dir, name)
    corpus[name[:3]] = Corpus_stat(path, sep="\t")

languages = corpus.keys()
print("Computing common GSW words")
gsw_words = corpus["gsw"].get_common_words(10000)

# Create the dataframe containing proportion of words for each language
df = pd.DataFrame(index=gsw_words)
print("Computing the proportions of words for each language")
for language in languages:
    print(language)
    df[language] = df.index.map(corpus[language].get_proportion)

# Load the speaker counts to weight the proportions
df_speakers = pd.read_csv(path_speakers, index_col="language_code")
for language in languages:
    if weighted:
        df[language] = df[language] * df_speakers["speakers"][language]

df["sum_ratio"] = df[languages].sum(axis=1)
df["gsw_ratio"] = df["gsw"] / df["sum_ratio"]

# Apply specificity on the ratio to filter out words that are not specific
# enough. Then we take the 400 most common words, which corresponds to the
# filtered words with the highest proportion in Swiss-German.
# Note that we should not simply sort by ratio descending and return
# the list because a high ratio does not always mean the word is frequent in
# Swiss-German. Think about a very rare word that appears only in Swiss-German.
# It would have a ratio of 1, which is the highest.

gsw_cov_list = []
german_cov_list = []
for specificity in specificities:
    print(f"Specificity : {specificity}")
    filtered = df[df["gsw_ratio"] > specificity]
    final = filtered["gsw"].sort_values(ascending=False)
    common_words = final.index[:400]
    if len(common_words) < 400:
        print("*"*79)
        print("WARNING : Not enough common words, you may need to increase " +
              "the amount of GSW words used in the dataframe")
        print("*"*79)
    gsw_coverage = corpus["gsw"].get_coverage(common_words)
    gsw_cov_list.append(gsw_coverage)
    print(f"GSW coverage : {gsw_coverage}")
    german_coverage = corpus["deu"].get_coverage(common_words)
    german_cov_list.append(german_coverage)
    print(f"German coverage : {german_coverage}")
    print("Most common words")
    for x in common_words[:10]:
        print(x)

print([round(x*100, 3) for x in gsw_cov_list])
print([round(x*100, 3) for x in german_cov_list])
