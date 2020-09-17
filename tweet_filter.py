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
import logging
import traceback
import gc
import math
from preprocessing.cleaner import *
from tqdm import tqdm
from pathlib import Path

# Define typing aliases
Coords = Tuple[float, float]
Sentences = List[Tuple[int, str]]
# Sentences_pred : elements are (idx, sentence, gsw_prediction)
Sentences_pred = List[Tuple[int, str, float]]
# Idx_to_location : key=idx, value = (coords object, (lon, lat), geo_source)
Idx_to_location = Dict[int, Tuple[Any, Coords, str]]
# GSW_tweets : elements are
# (sentence, (lon, lat), gsw prediction, geo_source, user_id, original tweet)
GSW_tweets = List[Tuple[str, Tuple[float, float], float, str, str, dict]]

class TweetFilter:
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

        print("Initializing...")
        self.config = load_yaml(config) if isinstance(config, str) else config
        # remove previous logging config if present
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # create new logging config
        path_log = os.path.join(self.config["dir_path_log"], "filter.log")
        logging.basicConfig(filename=path_log,
                            filemode='w',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        self.geocoder = Geocoder(self.config)
        self.filterer = PatternSentenceFilter()
        self.splitter = MocySplitter()
        # Set the gpu
        cuda.set_device(self.config["gpu_index_to_use"])
        self.lid = BertLid()
        self.tweets = None
        self.processed_tweets_ids = None
        self.new_tweets_ids = set()

        raw_tweets_stream_dir_path = self.config["raw_tweets_stream_dir_path"]
        raw_tweets_search_dir_path = self.config["raw_tweets_search_dir_path"]
        self.raw_tweets_paths = [(os.path.join(raw_tweets_stream_dir_path, x),
                                 "stream")
                                 for x in os.listdir(raw_tweets_stream_dir_path)
                                 if x[-4:] == ".txt"]
        self.raw_tweets_paths += [(os.path.join(raw_tweets_search_dir_path, x),
                                 "search")
                                 for x in os.listdir(raw_tweets_search_dir_path)
                                 if x[-4:] == ".txt"]

        # Create an empty file if the processed ids file does not exist
        if not os.path.exists(self.config["processed_tweets_ids_path"]):
            with open(self.config["processed_tweets_ids_path"], "w",
                      encoding="utf8"):
                pass

        Path(self.config["out_dir_tweet_processing"]).mkdir(parents=True,
                                                            exist_ok=True)

        with open(self.config["processed_tweets_ids_path"], "r",
                  encoding="utf8") as f:
            ids = f.readlines()
            ids = [id[:-1] if id[-1]=="\n" else id for id in ids]
            self.processed_tweets_ids = set(ids)

        if not os.path.exists(self.config["sg_users_count_path"]):
            df = pd.DataFrame([], columns=["user_id", "gsw_tweet_count"])
            df.set_index("user_id", inplace=True)
            df.to_csv(self.config["sg_users_count_path"])


    @accepts(Any, dict)
    @returns(bool)
    def is_geo_available(self, tweet) -> bool:
        """Return True if at least one of the following geographic information
        is available in the tweet :
            - Precise GPS coordinates of the location from where the tweet has
              been sent
            - Coordinates of a bounding box representing the place from where
              the tweet has been sent.
            - A not empty user.location field. This is a free text field given
              by the user on his twitter profile.
        """
        gps = bool(tweet.get("coordinates", None))
        place = bool(tweet.get("place", None))
        min_length = self.config["min_location_length"]
        location = "user" in tweet \
                    and bool(tweet["user"].get("location", None)) \
                    and len(tweet["user"]["location"]) >= min_length
        return gps or place or location

    @staticmethod
    @accepts(dict)
    @returns(List[dict])
    def _extract_sub_tweets_from_tweet(tweet: Dict) -> List[Dict]:
        """From a tweet, extract all tweets that are contained in it. There are
        two paths that may contain another tweet :
        1) _json/retweeted_status
        2) _json/quoted_status
        The function update the tweet list by adding all sub-tweets recursively.
        Note that the top-level tweet is not included in the sub-tweet if it is
        a retweet. This is because the text is exactly the same as the original
        tweet that we return anyway since it is a retweet.
        """

        sub_tweets = []
        retweet = False
        # Get the two potential subtweets
        if tweet.get("retweeted_status", None):
            sub_tweets.append(tweet["retweeted_status"])
            retweet = True
        if tweet.get("quoted_status", None):
            sub_tweets.append(tweet["quoted_status"])

        # Inspect the subtweets recursively
        all_sub_tweets = []
        for tweet in sub_tweets:
            new_tweets = TweetFilter._extract_sub_tweets_from_tweet(tweet)
            all_sub_tweets.extend(new_tweets)

        # If the current tweet is not a retweet, add it
        if not retweet:
            all_sub_tweets.append(tweet)

        return all_sub_tweets

    @staticmethod
    @accepts(List[dict])
    @returns(List[dict])
    def _extract_sub_tweets(tweets: List[Dict]) -> List[Dict]:
        """From the list of tweets, extract all tweets that are contained in
        each tweet.
        """

        all_sub_tweets = []
        for tweet in tweets:
            sub_tweets = TweetFilter._extract_sub_tweets_from_tweet(tweet)
            all_sub_tweets.extend(sub_tweets)
        return all_sub_tweets

    @accepts(Any, List[dict])
    @returns(List[dict])
    def _filter_out_duplicates(self, tweets: List[Dict]) -> List[Dict]:
        """Filter out duplicated tweets based on the tweet id. A tweet is
        considered a duplicate either if another tweet in the list has the same
        id, or if we already processed this tweet id in a previous batch"""

        filtered_tweets = []
        for tweet in tweets:
            if "id_str" in tweet \
            and not str(tweet["id_str"]) in self.processed_tweets_ids:
                self.processed_tweets_ids.add(str(tweet["id_str"]))
                self.new_tweets_ids.add(str(tweet["id_str"]))
                filtered_tweets.append(tweet)
        return filtered_tweets

    @accepts(Any, List[dict])
    @returns(List[dict])
    def _filter_geo(self, tweets: List[Dict]) -> List[Dict]:
        """Filter out tweets with no geographic data available"""
        return [tweet for tweet in tweets if self.is_geo_available(tweet)]

    @staticmethod
    @accepts(List[Dict])
    @returns(Sentences)
    def _extract_text_from_tweets(tweets: List[Dict]) -> Sentences:
        """Extract the text field of all given tweets. This function tries to
        extract the 'full_text' attribute, and take the 'text' attribute if not
        available, or raises an Exception if none are found.
        """

        if not isinstance(tweets, list):
            raise ValueError("'tweets' must be a list")
        sentences = []
        for i in range(len(tweets)):
            tweet = tweets[i]
            if "extended_tweet" in tweet \
            and "full_text" in tweet["extended_tweet"]:
                sentences.append((i, tweet["extended_tweet"]["full_text"]))
            elif "full_text" in tweet:
                sentences.append((i, tweet["full_text"]))
            elif "text" in tweet:
                sentences.append((i, tweet["text"]))
            else:
                # Some known tweets that are not statuses
                if len(tweet) == 1 and "limit" in tweet:
                    pass
                else:
                    # If the tweet type is unknown
                    print(tweet)
                    raise ValueError("Cannot retrieve text from tweet")
        return sentences

    @accepts(Any, Sentences)
    @returns(Sentences)
    def _preprocess(self, sentences: Sentences) -> Sentences:
        """Preprocess the text to remove special elements
        """
        print("    Specific twitter preprocessing")
        clean_texts = []
        for sentence in sentences:
            idx = sentence[0]
            text = sentence[1]
            # Specific twitter preprocessing (remove RT, MT, mentions, hashtags,
            # and urls)
            for regex, replacement in self.config["preprocessing_regex"]:
                text = re.sub(regex, replacement, text)
            clean_texts.append((idx, text))

        # remove hat elements
        print("    Removing hat elements")
        sentences = clean_texts.copy()
        clean_texts = []
        for sentence in sentences:
            idx = sentence[0]
            text = Cleaner.remove_hat_element(sentence[1])
            clean_texts.append((idx, text))

        # remove smileys
        print("    Removing smileys")
        sentences = clean_texts.copy()
        clean_texts = []
        for sentence in sentences:
            idx = sentence[0]
            text = Cleaner.remove_smileys(sentence[1])
            clean_texts.append((idx, text))

        # remove html entities
        print("    Removing html entities")
        sentences = clean_texts.copy()
        clean_texts = []
        for sentence in sentences:
            idx = sentence[0]
            text = Cleaner.remove_html_entities(sentence[1])
            clean_texts.append((idx, text))

        # uniformize punctuation
        print("    Uniformizing punctuation")
        sentences = clean_texts.copy()
        clean_texts = []
        for sentence in sentences:
            idx = sentence[0]
            text = Cleaner.clean_punc(sentence[1])
            clean_texts.append((idx, text))

        return clean_texts

    @staticmethod
    @accepts(Sentences)
    @returns(Sentences)
    def _normalize_texts(sentences: Sentences) -> Sentences:
        """Wrapper on 'normalize_text' that applies the function to all elements
        of a list"""

        normalized = []
        for sentence in sentences:
            idx = sentence[0]
            text = sentence[1]
            text = normalize_text(text, strip_emojis=True)
            normalized.append((idx, text))

        return normalized

    @accepts(Any, Sentences)
    @returns(Sentences)
    def _split_texts(self, sentences: Sentences) -> Sentences:
        """For each text of a list, split the text into sentences and associate
        each sentence to the index of the text. Return the list of tuple [index,
        sentence]"""

        splits = [(idx, self.splitter.split(text))
                  for idx, text in sentences]
        sentences = [(idx, text) for idx, split in splits
                     for text in split]
        return sentences

    @staticmethod
    @accepts(Sentences)
    @returns(Sentences)
    def _remove_sentences_with_special_chars(sentences: Sentences) -> Sentences:
        """Remove sentences that contains at least one very special characters
        """

        chars_ok = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        chars_ok += "ÀÁÂÄÈÉÊËÍÌÎÏÓÒÔÖÚÙÛÜàáâäèéêëìíîïôöòóüùúûÿ"
        chars_ok += " -,.?!0123456789%&\"\'()/$*+:;<=>[]\\^_{}|\\~€°²"
        chars_ok = set(chars_ok)

        return [x for x in sentences
                if len(set(x[1]).difference(chars_ok))==0]

    @staticmethod
    @accepts(Sentences, int)
    @returns(Sentences)
    def _remove_groups_of_special_chars(sentences: Sentences,
                                        from_size: int) -> Sentences:
        return [(idx, Cleaner.remove_groups_of_special_chars(text, from_size))
                for idx, text in sentences]

    @staticmethod
    @accepts(Sentences, int)
    @returns(Sentences)
    def _remove_sentences_with_special_words(sentences: Sentences,
                                             max_char: int) -> Sentences:
        """Remove sentences with words containing too much special characters.
        """

        chars_ok = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        chars_ok += "ÀÁÂÄÈÉÊËÍÌÎÏÓÒÔÖÚÙÛÜàáâäèéêëìíîïôöòóüùúûÿ"
        chars_ok += "0123456789"
        chars_ok = set(chars_ok)

        res = []
        for sentence in sentences:
            words = sentence[1].split()
            ok = True
            for word in words:
                if len([c for c in word if c not in chars_ok]) > max_char:
                    ok = False
            if ok:
                res.append(sentence)
        return res

    @staticmethod
    @accepts(Sentences)
    @returns(Sentences)
    def _remove_isolated_special_chars(sentences: Sentences) -> Sentences:
        return [(idx, Cleaner.remove_isolated_special_chars(text))
                for idx, text in sentences]


    @staticmethod
    @accepts(Sentences)
    @returns(Sentences)
    def _remove_special_duplication(sentences: Sentences) -> Sentences:
        return [(idx, Cleaner.remove_special_duplication(text))
                for idx, text in sentences]

    # @staticmethod
    # @accepts(Sentences)
    # @returns(Sentences)
    # def _remove_special_chars(sentences: Sentences) -> Sentences:
    #     """Remove groups of special characters from a list of sentences"""
    #
    #     threshold = self.config["max_consecutive_special_chars"] + 1
    #     return [Cleaner.remove_groups_of_special_chars(x, threshold)
    #             for x in sentences]

    # @staticmethod
    # @accepts(Sentences)
    # @returns(Sentences)
    # def _remove_special_chars(sentences: Sentences) -> Sentences:
    #     """Remove special characters from a list of sentences"""
    #
    #     chars_ok = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    #     chars_ok += "ÀÁÂÄÈÉÊËÍÌÎÏÓÒÔÖÚÙÛÜàáâäèéêëìíîïôöòóüùúûÿ"
    #     chars_ok += " -,.?!0123456789%&\"\'()/"
    #
    #     cleaned = []
    #     for sentence in sentences:
    #         idx = sentence[0]
    #         text = sentence[1]
    #
    #         chars = set(list(text))
    #         for c in chars:
    #             if not c in chars_ok:
    #                 text = text.replace(c, " ")
    #         text = re.sub(r"\s+", " ", text)
    #         text = re.sub(r"\s,", ",", text)
    #         text = text.strip()
    #         cleaned.append((idx, text))
    #
    #     return cleaned

    @accepts(Any, Sentences)
    @returns(Sentences)
    def _filter_valid_sentences(self, sentences: Sentences) -> Sentences:
        """Filter out all sentences that are not considered well-formed,
        according to rules defined in the corresponding yaml file. The method
        takes a list of tuple [index, sentence] as input and output the same
        list, with some elements filtered out.
        """
        return [(idx, text) for idx, text in sentences
                            if self.filterer.is_valid(text)]

    @accepts(Any, List[int])
    @returns(Idx_to_location)
    def _geocode_tweets(self, indices: List[int]) -> Idx_to_location:
        """From a list of indices, retrieve the location from the tweets and
        forward geocode the location. The method returns a dict mapping each
        index to a tuple [location, coordinates, type]
            location : location object, according to 'type'.
                - type==gps : the location will be a list of [Long, Lat]
                - type==place : the location will be a 'twitter place' object
                - type==location : forward geocoding from the user.location
                                   field of the tweet
            coordinates : gps coordinates (long, lat) derived from location
            type : the type of the location object
                - gps : precise gps coordinates from the tweet itself
                - twitter_place : a twitter place that include details such as
                                  loose coordinates (bounding box)
                - locationiq : forward geocoding from the user.location field of
                               the tweet using the locationIQ api.
        """
        self.geocoder.open_mapping_file()
        idx_to_location = dict()
        # Get the indices of all tweets we want to geocode.
        indices = set(indices)
        for idx in tqdm(indices):
            tweet = self.tweets[idx]
            if tweet.get("coordinates", None):
                location = tuple(tweet["coordinates"]["coordinates"])
                coordinates = (float(location[0]), float(location[1]))
                loc_type = "GPS"
                idx_to_location[idx] = (location, coordinates, loc_type)
            elif tweet.get("place", None) \
            and tweet["place"].get("bounding_box", None) \
            and tweet["place"]["bounding_box"].get("coordinates", None) \
            and len(tweet["place"]["bounding_box"]["coordinates"]) > 0:
                location = tweet["place"]
                polygon = tweet["place"]["bounding_box"]["coordinates"]
                if len(polygon) > 1:
                    print("Warning, multiple polygons to define a place " +
                    "(tweet id: " + str(tweet["id_str"]) + ")")
                    print(polygon)
                longitudes = [x[0] for x in polygon[0]]
                latitudes = [x[1] for x in polygon[0]]

                longitude = mean([min(longitudes), max(longitudes)])
                latitude = mean([min(latitudes), max(latitudes)])
                coordinates = (float(longitude), float(latitude))
                loc_type = "Twitter_place"
                idx_to_location[idx] = (location, coordinates, loc_type)
            elif "user" in tweet and "location" in tweet["user"]:
                location_str = tweet["user"]["location"]
                if location_str is None:
                    location_str = ""
                found = False
                if len(location_str) > 1:
                    try:
                        location = self.geocoder.forward_geocode(location_str)
                        if len(location[0].keys()) > 0:
                            coordinates = (float(location[0]["lon"]),
                                           float(location[0]["lat"]))
                            loc_type = location[1]
                            idx_to_location[idx] = (location, coordinates, loc_type)
                            found = True
                    except Exception:
                        print("*** Cannot forward geocode ***")
                        print(location_str)
                        print("---")
                        print(traceback.print_exc())
                        print("******************************")
                if not found:
                    idx_to_location[idx] = (None, (0.0,0.0), "")
            elif "user" in tweet:
                idx_to_location[idx] = (None, (0.0,0.0), "")


        self.geocoder.clean()
        return idx_to_location


    @accepts(Any, Sentences_pred, Idx_to_location, Any)
    @returns(GSW_tweets)
    def _attach_gsw_location(self,
                             sentences_pred: Sentences_pred,
                             idx_to_location: Idx_to_location,
                             keep_foreign=True):
        """From a list of Sentences_pred and a dictionary mapping an index to a
        location, attach locations to sentences. Optionally, filter out all
        locations that are not in switzerland if 'keep_foreign' == False.
        Returns the sentences with location attached.

        Returns
            # (sentence, (lon, lat), gsw prediction, geo_source, original tweet)
            List[Tuple[str, Tuple[float, float], float, str, str, dict]]
                A list where each element is a tuple containing 5 elements :
                    - The sentence
                    - The coordinates of the user location
                    - The gsw prediction
                    - The source of the coordinates, can be
                        * GPS : the original tweets contains GPS coordinates
                        * Twitter_place : the original tweets contains a twitter
                          place
                        * Geocoder_original : the coordinates were retrieved by
                          querying locationiq with the user.location field
                        * Geocoder_CH_word : the coordinates were retrievec by
                          querying locationiq with a CH word (city name, postal
                          code...) found in the user.location field
                        * "" : No location available
                    - The raw tweet object corresponding to the sentence
        """
        gsw_tweets = []
        for sentence_pred in sentences_pred:
            tweet_id = sentence_pred[0]
            coords = idx_to_location[tweet_id][1]
            geo_source = idx_to_location[tweet_id][2]
            if keep_foreign \
            or self.geocoder.are_coords_in_switzerland(coords):
                gsw_tweets.append((sentence_pred[1],
                                   coords,
                                   sentence_pred[2],
                                   geo_source,
                                   self.tweets[tweet_id]["user"]["id_str"],
                                   self.tweets[tweet_id]))

        return gsw_tweets

    @accepts(Any, Sentences)
    @returns(Sentences_pred)
    def _filter_gsw_sentences(self, sentences):
        """Filter out all sentences that are not detected as Swiss-German
        """
        # Predict Swiss-German
        sentences_list = [sentence[1] for sentence in sentences]
        predictions = []
        # separate in batches to avoid cuda out of memory error
        for i in tqdm(range(math.ceil(len(sentences_list)/100))):
            left = i*100
            right = (i+1)*100
            batch = sentences_list[left:right]
            if len(batch) > 0:
                predictions.extend(self.lid.predict_label(batch))
        if len(sentences_list) != len(predictions):
            raise Exception("predictions and sentences_list must have the " +
                            "same length")

        # Create the gsw_tweet object for each prediction that exceeds a
        # threshold
        filtered = []
        for i in range(len(sentences)):
            prediction = float(predictions[i])
            if prediction >= self.config["lid_threshold"]:
                filtered.append((sentences[i][0], sentences[i][1], prediction))

        del(predictions)
        gc.collect()
        cuda.empty_cache()
        return filtered

    @accepts(Any, GSW_tweets)
    @returns(GSW_tweets)
    def _remove_non_gsw_accent(self, gsw_tweets):
        return [(Cleaner.remove_non_gsw_accent(x[0]), x[1],x[2],x[3],x[4],x[5])
                for x in gsw_tweets]

    @accepts(Any, GSW_tweets)
    @returns(None)
    def _write_gsw_tweets(self, gsw_tweets):
        """Write the swiss-german tweets on disk as a pickle.
        """
        dir_path = self.config["out_dir_tweet_processing"]
        out_path = get_new_file_path(dir_path, ".pkl")

        save_obj(gsw_tweets, out_path)

        msg = "Writing " + str(len(gsw_tweets)) + " sentences to " + \
              str(out_path)
        logging.info(msg)

    @accepts(Any, GSW_tweets)
    @returns(int)
    def _write_new_sg_users(self, gsw_tweets):
        """Write the newly found Swiss-German users on disk and update the
        counts of gsw sentences for each twitter user.

        Returns the count of new Swiss-German twitter user found. A user is
        added if at least one of his tweet is Swiss-German with a very high
        probability.
        """
        df = pd.read_csv(self.config["sg_users_count_path"],
                         index_col="user_id")
        df.index = df.index.map(str)
        new_users_count = 0
        for gsw_tweet in gsw_tweets:
             prediction = gsw_tweet[2]
             current_user_id = gsw_tweet[4]
             if prediction >= self.config["threshold_new_sg_user"]:
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
        with open(self.config["processed_tweets_ids_path"], "a", encoding="utf8") as f:
            for id in self.new_tweets_ids:
                f.write(str(id) + "\n")
        self.new_tweets_ids = set()

    def process(self, cur_gsw_fetched):
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

        msg = "GSW sentences fetched from stream : " + \
              str(cur_gsw_fetched["stream"])
        print(msg)
        logging.info(msg)
        msg = "GSW sentences fetched from search : " + \
              str(cur_gsw_fetched["search"])
        print(msg)
        logging.info(msg)

        try:
            paths_used = []
            for path, source in self.raw_tweets_paths:
                with open(path, "r", encoding="utf8") as f:
            #for path, source in self.raw_tweets_paths:
            #    with open(path, "r", encoding="utf8") as f:
                    print("Loading " + path + "...")
                    raw_tweets = f.readlines()
                    raw_tweets = [x for x in raw_tweets if x != '\n']
                    #self.tweets = [json.loads(x) for x in raw_tweets]
                    tmp = []
                    for x in raw_tweets:
                        try:
                            tmp.append(json.loads(x))
                        except Exception:
                            print("*********************")
                            print(x)
                            print("*********************")
                    self.tweets = tmp
                    print("Processing...")
                    print("  => " + str(len(self.tweets)) + " raw tweets")
                    self.tweets = TweetFilter._extract_sub_tweets(self.tweets)
                    print("  => " + str(len(self.tweets)) + " sub-tweets")
                    self.tweets = self._filter_out_duplicates(self.tweets)
                    print("  => " + str(len(self.tweets)) + " unique tweets " +
                        " not already processed")
                    # Extract the text for each tweet. At this point 'sentences'
                    # represents a list of tuple, each tuple containing the index of
                    # the tweet (i.e. index of self.tweets) and the corresponding
                    # text
                    print("Extract text from tweets")
                    sentences = TweetFilter._extract_text_from_tweets(
                                                                    self.tweets)
                    print("Preprocessing text")
                    sentences = self._preprocess(sentences)

                    print("Normalizing text")
                    sentences = TweetFilter._normalize_texts(sentences)

                    print("Splitting text")
                    sentences = self._split_texts(sentences)
                    print(f"=> {len(sentences)} sentences")

                    print("Removing sentences that contain at least one very " +
                          "special character")
                    f = TweetFilter._remove_sentences_with_special_chars
                    sentences = f(sentences)
                    print(f"=> {len(sentences)} sentences")

                    print("Removing words that are composed only of special " +
                          "chars")
                    f = TweetFilter._remove_groups_of_special_chars
                    sentences = f(sentences,
                                  self.config["min_char_special_group"])

                    print("Removing sentences containing words with too much " +
                          "special characters")
                    f = self._remove_sentences_with_special_words
                    max_char = self.config["max_special_char_in_word"]
                    sentences = f(sentences, max_char)
                    print(f"  => {len(sentences)} sentences")

                    print("Removing duplication of special characters")
                    f = TweetFilter._remove_special_duplication
                    sentences = f(sentences)

                    print("Removing isolated special chars")
                    f = TweetFilter._remove_isolated_special_chars
                    sentences = f(sentences)

                    print("Filtering valid sentences")
                    sentences = self._filter_valid_sentences(sentences)
                    print(f"  => {len(sentences)} well formed sentences")

                    print("Filtering gsw...")
                    # sentences_pred: elements are (idx, sentence, prediction)
                    sentences_pred = self._filter_gsw_sentences(sentences)
                    print(f"  => {len(sentences_pred)} gsw sentences found")
                    cur_gsw_fetched[source] += len(sentences_pred)

                    print("Geocoding...")
                    indices = [x[0] for x in sentences_pred]
                    idx_to_location = self._geocode_tweets(indices)
                    gsw_tweets = self._attach_gsw_location(
                                        sentences_pred,
                                        idx_to_location,
                                        self.config["keep_foreign_location"])
                    if not self.config["keep_foreign_location"]:
                        print(f"  => {len(gsw_tweets)} sentences " +
                              "geolocalized in Switzerland")

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


                os.remove(path)

                msg = "GSW sentences fetched from stream : " + \
                      str(cur_gsw_fetched["stream"])
                print(msg)
                logging.info(msg)
                msg = "GSW sentences fetched from search : " + \
                      str(cur_gsw_fetched["search"])
                print(msg)
                logging.info(msg)
        except Exception:
            traceback.print_exc()
            logging.error(traceback.format_exc())


        msg = "GSW sentences fetched from stream : " + \
              str(cur_gsw_fetched["stream"])
        print(msg)
        logging.info(msg)
        msg = "GSW sentences fetched from search : " + \
              str(cur_gsw_fetched["search"])
        print(msg)
        logging.info(msg)

        print("\nAll files have been processed\n")
