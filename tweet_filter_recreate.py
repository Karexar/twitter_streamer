# This scripts is very similar to the tweet_filter scripts, except that it is
# designed to recreate the entire dataset using the raw tweets saved. This is
# used if we want a different preprocessing for the already fetched tweets.
# The static method from TweetFilter are used, and the non static method are
# modified here

import re
import json
from phrasal.norm_punc import normalize_text
from phrasal.pattern_sentence_filter import PatternSentenceFilter
from phrasal.mocy_splitter import MocySplitter
from utils.utils import *
from geocoder import *
from typing import List, Dict, Tuple, Union, Any
from bert_lid import BertLid
import os
from typechecker.typecheck import *
from statistics import mean
from torch import cuda
import pandas as pd
import traceback
import gc
import math
from preprocessing.cleaner import *
from tqdm import tqdm
from tweet_filter import *

# Define typing aliases
Coords = Tuple[float, float]
Sentences = List[Tuple[int, str]]
# Sentences_info : elements are (idx, sentence, (lon, lat), geo_source)
Sentences_info = List[Tuple[int, str, Tuple[float, float], str]]
# Idx_to_location : key=idx, value = (coords object, (lon, lat), geo_source)
Idx_to_location = Dict[int, Tuple[Any, Coords, str]]
# GSW_tweets : elements are
# (sentence, (lon, lat), gsw prediction, geo_source, original tweet)
GSW_tweets = List[Tuple[str, Tuple[float, float], float, str, dict]]

class TweetFilterRecreate(TweetFilter):
    """This class handles all the pipeline of tweet processing, that is loading,
    cleaning, geocoding, and filtering the tweets.

    The main function is 'process' that will apply the whole pipeline to the
    tweets.
    """

    @accepts(Any, Union[str, dict])
    @returns(None)
    def __init__(self, config: Union[str, dict]):
        """Load the tweets and initialize the objects to process them.
        """

        super().__init__()

        # Create an empty dataset of users and save it (will overwrite the
        # previous one)
        df = pd.DataFrame([], columns=["user_id", "gsw_tweet_count"])
        df.set_index("user_id", inplace=True)
        df.to_csv(self.config["sg_users_count_path"])

    @accepts(Any, List[dict])
    @returns(List[dict])
    def _filter_out_duplicates(self, tweets: List[Dict]) -> List[Dict]:
        """Filter out duplicates from the tweet list based on the tweet id.
        Here we only verify if the tweet appears in the new_tweets_ids, not
        in the processed_tweets_id, since we want to reprocess them all"""

        filtered_tweets = []
        for tweet in tqdm(tweets):
            if not str(tweet["id_str"]) in self.new_tweets_ids:
                self.new_tweets_ids.add(str(tweet["id_str"]))
                filtered_tweets.append(tweet)
        return filtered_tweets

    @accepts(Any, GSW_tweets)
    @returns(None)
    def _write_gsw_tweets(self, gsw_tweets):
        """Write the swiss-german tweets on disk as a pickle.
        """
        save_obj(gsw_tweets, self.config["path_dirty_gsw_tweets"])

    @accepts(Any, GSW_tweets)
    @returns(int)
    def _write_new_sg_users(self, gsw_tweets):
        """Write the newly found Swiss-German users on disk and update the
        counts of gsw sentences for each twitter user.
        """
        df = pd.read_csv(self.config["sg_users_count_path"],
                         index_col="user_id")
        user_ids = set(df.index)
        new_users_count = 0
        for gsw_tweet in gsw_tweets:
            user_id = gsw_tweet[4]
            prediction = gsw_tweet[2]
            if current_user_id in df.index:
                df.at[current_user_id, "gsw_tweet_count"] += 1
            else:
                df.loc[current_user_id, "gsw_tweet_count"] = 1
                new_users_count += 1
        df.to_csv(self.config["sg_users_count_path"])
        return new_users_count

    @accepts(Any)
    @returns(None)
    def _update_processed_tweets(self):
        """Update the processed tweets ids."""
        with open(self.config["processed_tweets_ids_path"], "w",
                  encoding="utf8") as f:
            new_ids = set(self.new_tweets_ids)
            merged_ids = self.processed_tweets_ids.union(new_ids)
            for id in merged_ids:
                f.write(str(id) + "\n")

    def process(self, path):
        """Process all tweets according to the pipeline :
        1. Extract sub-tweets
        2. Filter out duplicates and tweets that have already been processed
        3. Extract text
        4. Process text
        5. Normalize text
        6. Split text into sentences
        7. Remove special characters
        8. Keep well-formed sentences
        9. Forward geocode
        10. Attach Swiss-german location
        11. Filter Swiss-german language
        12. Write the gsw tweets on disk
        13. Update the Swiss-German twitter users
        14. Update the processed tweets ids
        """

        with open(path, "r", encoding="utf8") as f:
            print("Loading " + path + "...")
            tweets = load_obj(path)
            self.tweets = [json.loads(x[4]) for x in tweets]

            print(f"Processing {len(self.tweets)} already fetched tweets...")
            print("Extracting sub-tweets")
            self.tweets = TweetFilter._extract_sub_tweets(self.tweets)
            print("  => " + str(len(self.tweets)) + " sub-tweets")
            print("Filtering out duplicates")
            self.tweets = self._filter_out_duplicates(self.tweets)
            print("  => " + str(len(self.tweets)) + " unique tweets ")
            # Extract the text for each tweet. At this point 'sentences'
            # represents a list of tuple, each tuple containing the index of
            # the tweet (i.e. index of self.tweets) and the corresponding
            # text
            print("Extracting text from tweets")
            sentences = TweetFilter._extract_text_from_tweets(self.tweets)

            print("Preprocessing text")
            sentences = self._preprocess(sentences)

            print("Normalizing text")
            sentences = TweetFilter._normalize_texts(sentences)

            print("Splitting the text")
            sentences = self._split_texts(sentences)
            print(f"=> {len(sentences)} sentences")

            print("Removing sentences that contain at least one very special character")
            sentences = TweetFilter._remove_sentences_with_special_chars(sentences)
            print(f"=> {len(sentences)} sentences")

            print("Removing words that are composed only of special chars")
            sentences = TweetFilter._remove_groups_of_special_chars(sentences, self.config["min_char_special_group"])

            print("Removing sentences containing words with too much special characters")
            sentences = self._remove_sentences_with_special_words(sentences)
            print(f"=> {len(sentences)} sentences")

            print("Removing duplication of special characters")
            sentences = TweetFilter._remove_special_duplication(sentences)

            print("Removing isolated special chars")
            sentences = TweetFilter._remove_isolated_special_chars(sentences)

            print("Filtering valid sentences")
            sentences = self._filter_valid_sentences(sentences)
            print(f"  => {len(sentences)} well formed sentences")

            print("Geocoding...")
            indices = [x[0] for x in sentences]
            idx_to_location = self._geocode_tweets(indices)
            sentences_info = self._attach_gsw_location(
                                sentences,
                                idx_to_location,
                                self.config["keep_foreign_location"])
            if not self.config["keep_foreign_location"]:
                print(f"  => {len(sentences_info)} sentences " +
                      "geolocalized in Switzerland")

            print("Filtering gsw...")
            gsw_tweets = self._filter_gsw_sentences(sentences_info)
            print(f"  => {len(gsw_tweets)} gsw sentences were found")

            print("Removing non gsw accents")
            gsw_tweets = self._remove_non_gsw_accent(gsw_tweets)

            print("Writing gsw tweets on disk...")
            self._write_gsw_tweets(gsw_tweets)

            print("Writing Swiss-German twitter users...")
            count = self._write_new_sg_users(gsw_tweets)
            print(f"  => {count} new Swiss-German users found")

            print("Updating processed tweets ids")
            self._update_processed_tweets()

            print("Done")

        print("\nAll files have been processed\n")
