# This script computes the 400 most common words in Swiss-German, print the
# 10 most common words, and print the coverage on a GSW and German corpus.
# It does the same with the 400 most common words specific to GSW (i.e. that
# do not appear in the german corpus)

from twitter.corpus_stat import *
import pandas as pd

###  Settings  ################################################################
gsw_path = "data/swisstext_leipzig/swisstext_leipzig.txt"
de_path = "data/leipzig_32/deu-de_web-public_2019_1M_converted.txt"
###############################################################################

def print_coverage(corpus_gsw, corpus_de, words):
    print("Computing coverages...")
    print(f"GSW coverage : {corpus_gsw.get_coverage(words)}")
    print(f"German coverage : {corpus_de.get_coverage(words)}")

print("Loading GSW corpus...")
corpus_gsw = Corpus_stat(gsw_path, sep="\t")
print("Loading German corpus...")
corpus_de = Corpus_stat(de_path, sep="\t")

print("Computing most common GSW words...")
common_gsw_words = corpus_gsw.get_common_words(400)
print("Most common GSW words :")
for x in common_gsw_words[:10]:
    print("  " + x)

print_coverage(corpus_gsw, corpus_de, common_gsw_words)

all_words_de = corpus_de.get_vocabulary()
all_common_gsw_words = pd.Series(corpus_gsw.get_common_words())
gsw_words_exclusive = set(all_common_gsw_words).difference(all_words_de)
mask = all_common_gsw_words.isin(gsw_words_exclusive)
gsw_words_exclusive = all_common_gsw_words[mask][:400]

print("Most common GSW words that don't appear in German :")
for x in gsw_words_exclusive[:10]:
    print("  " + x)

print_coverage(corpus_gsw, corpus_de, gsw_words_exclusive)
