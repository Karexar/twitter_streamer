# This script is used to clean the twitter dataset from unrelevant characters.
# However, a filter has been added to filter.py to take care of that, so this
# script should not be needed in the future.
# It reads the pickle file, clean the sentences, update the lid prediction,
# modify the .csv file accordingly, and save both file on disk

import pandas as pd
from utils.utils import *
import re
from tqdm import tqdm
from bert_lid import BertLid
import _thread
import os

###  Settings  ################################################################
dir_path = "twitter/final_dataset"
lid_threshold = 0.9
###############################################################################

_thread.start_new_thread( keep_alive, tuple() )

print("Loading...")
tweets = load_obj(os.path.join(dir_path, "gsw_tweets.pkl"))
print(f"{len(tweets)} sentences")

chars_ok = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
chars_ok += "ÀÁÂÄÈÉÊËÍÌÎÏÓÒÔÖÚÙÛÜàáâäèéêëìíîïôöòóüùúûÿ"
chars_ok += " -,.?!0123456789%&\"\'()/"

def clean_chars(sentence):
    chars = set(list(sentence))
    sentence = sentence.replace("ß", "ss")
    for c in chars:
        if not c in chars_ok:
            sentence = sentence.replace(c, " ")
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"\s,", ",", sentence)
    return sentence.strip()

print("Cleaning...")
cleaned = []
for t in tqdm(tweets):
    cleaned.append(clean_chars(t[0]))

print("Updating lid prediction...")
lid = BertLid()
preds = lid.predict_label(cleaned)

print("Filter out low gsw prediction score...")
final = []
drop = []
for i in range(len(preds)):
    if preds[i] >= lid_threshold:
        final.append((cleaned[i], tweets[i][1], preds[i], tweets[i][3], tweets[i][4]))
    else:
        drop.append((cleaned[i], tweets[i][1], preds[i], tweets[i][3], tweets[i][4]))
print(f"{len(final)} sentences with lid prediction >= {lid_threshold}")
print("Saving...")
save_obj(final, os.path.join(dir_path, "gsw_tweets2.pkl"))
save_obj(drop, os.path.join(dir_path, "gsw_drop.pkl"))

res = [[x[0], x[1], x[2], x[3], x[4]["id_str"]] for x in final]
df = pd.DataFrame(res, columns=["sentence",
                                "coords",
                                "gsw_prediction",
                                "geo_source",
                                "tweet_id"])
df.to_csv(os.path.join(dir_path, "gsw_sentences2.csv"), index=False)
