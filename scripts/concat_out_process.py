#import _thread
import os
from utils.utils import *
import pandas as pd
from tqdm import tqdm
import time
import datetime
from pathlib import Path
import os

###  Settings  ################################################################
dataset_dir = "dirty_dataset"
###############################################################################

def main():
    #_thread.start_new_thread( keep_alive, tuple() )

    config = load_yaml("config.yaml")
    dir_path = config["out_dir_tweet_processing"]

    print(datetime.datetime.now())
    print("Loading current dataset...")

    current_final_tweets = []
    if os.path.exists(config["path_dirty_gsw_tweets"]):
        current_final_tweets = load_obj(config["path_dirty_gsw_tweets"])

    cur_sentences = pd.DataFrame(columns=['sentence',
                                          'coords',
                                          'gsw_prediction',
                                          'geo_source',
                                          'user_id',
                                          'tweet_id'])
    if os.path.exists(config["path_dirty_gsw_sentences"]):
        cur_sentences = pd.read_csv(config["path_dirty_gsw_sentences"])

    count90 = cur_sentences[cur_sentences["gsw_prediction"] >= 0.9].shape[0]
    count95 = cur_sentences[cur_sentences["gsw_prediction"] >= 0.95].shape[0]
    count99 = cur_sentences[cur_sentences["gsw_prediction"] >= 0.99].shape[0]
    print(f"{count90} gsw sentences with prediction 0.9")
    print(f"{count95} gsw sentences with prediction 0.95")
    print(f"{count99} gsw sentences with prediction 0.99")
    assert(len(current_final_tweets) == len(cur_sentences))
    print("Adding new sentences...")
    ids_done = set(cur_sentences.tweet_id.apply(str).values)

    res = []
    full_object = []

    skip_count = 0
    names = [x for x in sorted(os.listdir(dir_path)) if x[-4:] == ".pkl"]
    for name in tqdm(names):
        path = os.path.join(dir_path, name)
        gsw_tweets = load_obj(path)
        # Filter out tweets already processed
        # This is useful only if for some reason we reset the processed_ids
        # file, otherwise the file ensures none of the tweets we are
        # adding has been already processed.
        filtered_tweets = []
        for t in gsw_tweets:
            if not str(t[5]["id_str"]) in ids_done:
                filtered_tweets.append(t)
            else:
                skip_count += 1

        full_object += filtered_tweets
        gsw_tweets = [[x[0],x[1],x[2],x[3],x[4],x[5]["id_str"]]
                      for x in filtered_tweets]
        res += gsw_tweets
    print(f"{skip_count} sentences already processed")

    df = pd.DataFrame(res, columns=["sentence",
                                    "coords",
                                    "gsw_prediction",
                                    "geo_source",
                                    "user_id",
                                    "tweet_id"])
    # concatenate with the already existing dataset
    df = pd.concat([cur_sentences, df])
    full_object = current_final_tweets + full_object

    assert(df.shape[0] == len(full_object))

    print("Saving back-up on disk...")
    Path(os.path.join(dataset_dir, "back_up")).mkdir(parents=True, exist_ok=True)

    value = str(round(time.time()))
    path = os.path.join(dataset_dir, f"back_up/gsw_sentences_{value}.csv")
    df.to_csv(path, index=False)
    path = os.path.join(dataset_dir, f"back_up/gsw_tweets_{value}.pkl")
    save_obj(full_object, path)
    print("Saving final dataset on disk...")
    path = os.path.join(dataset_dir, "gsw_sentences.csv")
    df.to_csv(path, index=False)
    path = os.path.join(dataset_dir, "gsw_tweets.pkl")
    save_obj(full_object, path)
    print(str(df.shape[0]) + " gsw sentences with prediction 0.9")
    length95 = df[df["gsw_prediction"] >= 0.95].shape[0]
    print(str(length95) + " gsw sentences with prediction 0.95")
    length99 = df[df["gsw_prediction"] >= 0.99].shape[0]
    print(str(length99) + " gsw sentences with prediction 0.99")
    df_filter = df.drop_duplicates(subset="sentence", keep="first")
    print(str(df.shape[0] - df_filter.shape[0]) + " duplicates")

    print("Removing temporary files...")
    for name in names:
        path = os.path.join(dir_path, name)
        #print(f"Removing {path}")
        os.remove(path)

if __name__ == "__main__":
    main()
