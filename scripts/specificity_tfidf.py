# This script computes the most common words according the TF-IDF scores of
# Swiss-German words with respect to 31 other languages. It computes the
# coverage over the GSW and German dataset for different value of the
# specificity parameter. It does the same by weighting the TF-IDF scores by
# the count of speaker for each languge.

from utils.utils import *
from corpus_class.corpus_32 import *
from corpus_class.corpus_stat import *
import pandas as pd
import _thread

###  Settings  ################################################################
leipzig_32_dir = "data/leipzig_32"
speaker_count = "data/speakers.csv"
gsw_path = "data/leipzig_32/gsw_swisstext_leipzig.txt"
deu_path = "data/leipzig_32/deu-de_web-public_2019_1M_converted.txt"
intervals = [round(x*0.05, 2) for x in range(17)]
intervals_pond = [round(x*0.005, 2) for x in range(201)]
# set langs to None to take all corpus in leipzig_32_dir
langs = ["gsw", "deu", "bar", "frr", "ksh", "lim", "nds", "pfl"]
###############################################################################

_thread.start_new_thread( keep_alive, tuple() )

print("Loading 32 language corpus...")
corpus = Corpus_32(leipzig_32_dir, speaker_count, langs)

print("Most common words using TF-IDF scores")
words_s0 = corpus.get_language_specific_words("gsw", 0.0, 400, False)
for x in words_s0[:10]:
    print(x)

print("Most common words using weighted TF-IDF scores and specificity=0.15")
for x in corpus.get_language_specific_words("gsw", 0.15, 400, True)[:10]:
    print(x)

print("Loading GSW corpus...")
corpus_gsw = Corpus_stat(gsw_path, sep="\t")
print("Loading German corpus...")
corpus_deu = Corpus_stat(deu_path, sep="\t")

gsw_coverage = corpus_gsw.get_coverage(words_s0)
print(f"  GSW coverage s=0 : {gsw_coverage}")
deu_coverage = corpus_deu.get_coverage(words_s0)
print(f"  German coverage s=0 : {deu_coverage}")
gsw_coverage = corpus_gsw.get_coverage(words_s002_pond)
print(f"  GSW coverage s=0.02 weighted: {gsw_coverage}")
deu_coverage = corpus_deu.get_coverage(words_s002_pond)
print(f"  German coverage s=0.02 weighted: {deu_coverage}")

# cov_gsw = []
# cov_deu = []
# print("Without ponderation")
# for specificity in intervals:
#     print(f"Specificity : {specificity}")
#     words = corpus.get_language_specific_words("gsw", specificity, 400)
#     gsw_coverage = corpus_gsw.get_coverage(words)
#     print(f"  GSW coverage : {gsw_coverage}")
#     deu_coverage = corpus_deu.get_coverage(words)
#     print(f"  German coverage : {deu_coverage}")
#     cov_gsw.append(gsw_coverage)
#     cov_deu.append(deu_coverage)
# print([round(x*100, 2) for x in cov_gsw])
# print([round(x*100, 2) for x in cov_deu])

cov_gsw_pond = []
cov_deu_pond = []
print("With ponderation")
for specificity in intervals_pond:
    print(f"Specificity : {specificity}")
    words = corpus.get_language_specific_words("gsw", specificity, 400, True)
    gsw_coverage = corpus_gsw.get_coverage(words)
    print(f"  GSW coverage : {gsw_coverage}")
    deu_coverage = corpus_deu.get_coverage(words)
    print(f"  German coverage : {deu_coverage}")
    cov_gsw_pond.append(gsw_coverage)
    cov_deu_pond.append(deu_coverage)
print([round(x*100, 3) for x in cov_gsw_pond])
print([round(x*100, 3) for x in cov_deu_pond])
