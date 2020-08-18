from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream, API, Cursor
import tweepy
from utils.utils import *
from corpus_class.corpus_stat import *
from corpus_class.corpus_32 import *
from corpus_class.corpus_8 import *
import json
import os
import traceback
from tweet_filter import *
import pandas as pd
import sys
import time
import logging
import socket
from urllib3.exceptions import ProtocolError, ReadTimeoutError

class StdOutListener(StreamListener):
    def __init__(self, config):
        super(StdOutListener, self).__init__()
        self.config = config
        self.dir_path_stream = self.config["raw_tweets_stream_dir_path"]
        self.dir_path_search = self.config["raw_tweets_search_dir_path"]
        self.out_file = None
        self.out_path = None
        self.count = 0

    def on_data(self, data):
        """ Write the whole tweet on disk. If self.filter_geo is True, we
        write the tweet only if geographic information is available.
        """
        try:
            #tweet = json.loads(data)
            self.out_file.write(data)
            self.count+=1
            if self.count >= self.config["raw_tweets_stream_batch_size"]:
                msg = "Writing " + str(self.count) + " tweets to " + \
                      str(self.out_file.name) + ".txt"
                logging.info(msg)
                print(msg)
                self.out_file.close()
                os.rename(self.out_path, self.out_path + ".txt")
                new_file_path = get_new_file_path(self.dir_path_stream, ".txt")
                self.out_path = new_file_path[:-4]
                self.out_file = open(self.out_path, 'w', encoding="utf8")
                self.count = 0
        except Exception:
            traceback.print_exc()

    def on_error(self, status):
        print(status)

class GSW_stream:
    """
    Stream for tweets in the switzerland area and filter
    them to keep well-formed Swiss-German sentences with
    a geo-localization.
    """

    def __init__(self, config_path):
        self.config = load_yaml(config_path)

        self.authentify_twitter()

        self.filter_languages = self.config["filter_languages"]
        if self.filter_languages == [] or self.filter_languages == [None]:
            self.filter_languages = None

        # Create the file containing gsw tweet count for each twitter user
        # if it does not exist
        if not os.path.exists(self.config["sg_users_count_path"]):
            df = pd.DataFrame([], columns=["user_id", "gsw_tweet_count"])
            df.set_index("user_id", inplace=True)
            df.to_csv(self.config["sg_users_count_path"])
        # Same for the file containing last tweet for each twitter user
        if not os.path.exists(self.config["sg_users_last_path"]):
            df = pd.DataFrame([], columns=["user_id", "last_tweet_id"])
            df.set_index("user_id", inplace=True)
            df.to_csv(self.config["sg_users_last_path"])

        print("Preparing the track words...")
        track_word_dir_path = self.config["track_word_dir_path"]
        if self.config["track_word_type"] == "common":
            # First check if we already have the data
            track_word_count = self.config["track_word_count"]
            track_word_file_name = "common_" + str(track_word_count) + ".pkl"
            path = os.path.join(track_word_dir_path, track_word_file_name)
            if track_word_file_name in os.listdir(track_word_dir_path):
                print("Track word data found, loading the file...")
                self.track_words = load_obj(path)
            else:
                print("Track word data not found, creating the data...")
                corpus_stat = Corpus_stat(self.config["gsw_corpus_path"])
                words = corpus_stat.get_common_words(track_word_count)
                save_obj(words, path)
                self.track_words = words
        elif self.config["track_word_type"] == "tfidf":
            track_word_count = self.config["track_word_count"]
            language = self.config["track_word_language"]
            specificity = self.config["track_word_specificity"]
            ponderation = self.config["track_word_ponderated"]
            track_word_file_name = "specific_" \
                                    + str(track_word_count) + "_" \
                                    + str(specificity)
            if ponderation:
                track_word_file_name = track_word_file_name + "_pond"
            track_word_file_name = track_word_file_name + ".pkl"

            # Checking if the file already exists
            path = os.path.join(track_word_dir_path, track_word_file_name)
            if track_word_file_name in os.listdir(track_word_dir_path):
                print("Track word data found, loading the file...")
                self.track_words = load_obj(path)
            else:
                print("Track word data not found, creating the data...")
                corpus_32 = Corpus_32(self.config["corpus_32_dir_path"],
                                      self.config["path_speakers"])
                words = corpus_32.get_language_specific_words(language,
                                                              specificity,
                                                              track_word_count,
                                                              ponderation)
                save_obj(words, path)
                self.track_words = words
        elif self.config["track_word_type"] == "proportion":
            track_word_count = self.config["track_word_count"]
            specificity = self.config["track_word_specificity"]
            ponderation = self.config["track_word_ponderated"]
            track_word_file_name = "proportion_" \
                                    + str(track_word_count) + "_" \
                                    + str(specificity)
            if ponderation:
                track_word_file_name = track_word_file_name + "_pond"
            track_word_file_name = track_word_file_name + ".pkl"

            # Checking if the file already exists
            path = os.path.join(track_word_dir_path, track_word_file_name)
            if track_word_file_name in os.listdir(track_word_dir_path):
                print("Track word data found, loading the file...")
                self.track_words = load_obj(path)
            else:
                print("Track word data not found, creating the data...")
                corpus_8 = Corpus_8(self.config["corpus_8_dir_path"])
                path_speakers = None
                if ponderation:
                    path_speakers = self.config["path_speakers"]
                words = corpus_8.get_gsw_specific_words(specificity,
                                                        track_word_count,
                                                        path_speakers)
                save_obj(words, path)
                self.track_words = words
        else:
            raise Exception("Wrong track_word_type value. Have '" \
                            + str(self.config["track_word_type"])
                            + "', expected either 'common' or 'specific'")

        print("Initialization done")

    def authentify_twitter(self):
        credentials = load_yaml(self.config["credentials_path"])

        consumer_key = credentials["twitter_api"]["consumer_key"]
        consumer_secret = credentials["twitter_api"]["consumer_secret"]
        access_token = credentials["twitter_api"]["access_token"]
        access_token_secret = credentials["twitter_api"]["access_token_secret"]

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = API(auth,
                       wait_on_rate_limit = True,
                       wait_on_rate_limit_notify= True)

        self.listener = StdOutListener(self.config)
        self.stream_obj = Stream(auth,
                                 self.listener,
                                 wait_on_rate_limit=True,
                                 wait_on_rate_limit_notify=True)

        try:
            self.api.verify_credentials()
            print("Authentication OK")
        except Exception:
            traceback.print_exc()

    def stream(self):
        path_log = os.path.join(self.config["dir_path_log"], "stream.log")
        create_logging_config(path_log)

        keep_going = True
        while keep_going:
            try:
                #self.stream_obj.filter(locations=[7.0, 45.915, 8.173, 47.621,
                #                              8.173, 46.214, 10.488, 47.81])
                dir_path = self.listener.dir_path_stream
                new_file_path = get_new_file_path(dir_path, ".txt")
                # We remove the extension such that the filter process does not
                # try to read it before the file is closed. The extension will
                # be added after the file is closed by renaming the file.
                self.listener.out_path = new_file_path[:-4]
                self.listener.out_file = open(self.listener.out_path, 'w',
                                              encoding="utf8")
                self.stream_obj.filter(languages=self.filter_languages,
                                       track=self.track_words)
            except KeyboardInterrupt:
                print("Interrupting streaming...")
                self.listener.out_file.close()
                os.rename(self.listener.out_path,
                          self.listener.out_path + ".txt")
                keep_going = False
            except (ValueError, ProtocolError, ReadTimeoutError,
                    socket.timeout):
                # This is to keep streaming even if we have an incomplete read
                # error
                traceback.print_exc()
                print("Retry streaming...")
                logging.exception("")
                time.sleep(10)
            except (socket.gaierror,
                    urllib3.exceptions.NewConnectionError,
                    urllib3.exceptions.MaxRetryError,
                    requests.exceptions.ConnectionError) as e:
                traceback.print_exc()
                print("Retry streaming...")
                logging.exception("")
                time.sleep(300)
            except Exception:
                traceback.print_exc()
                self.listener.out_file.close()
                os.rename(self.listener.out_path,
                          self.listener.out_path + ".txt")
                logging.exception("")
                raise

    @accepts(Any, list)
    @returns(None)
    def _write_tweets(self, tweets):
        """Write a list of raw tweets on disk"""
        # Get the most recent index
        out_path = get_new_file_path(self.listener.dir_path_search, ".txt")
        with open(out_path, "w", encoding="utf8") as f:
            for tweet in tweets:
                f.write(tweet + "\n")
            msg = "Writing " + str(len(tweets)) + " tweets to " + str(f.name)
            logging.info(msg)

    @accepts(Any, pd.core.frame.DataFrame, pd.core.frame.DataFrame)
    @returns(None)
    def update_sg_users_last(self, df_count, df_last):
        """Take all users that appears in df_count but not in df_last and add
        them to df_last. Save the resulting dataframe on disk
        """
        user_ids_count = set(df_count.index)
        user_ids_last = set(df_last.index)
        for idx in user_ids_count:
            if not idx in user_ids_last:
                df_last.loc[idx] = ["1"]
        df_last.to_csv(self.config["sg_users_last_path"])

    @accepts(Any, str, str, List[str], Dict[str, str], pd.core.frame.DataFrame)
    @returns(None)
    def search_user(self,
                    user_id,
                    last_tweet_id,
                    tweets,
                    update_last_tweet,
                    df_last):
        """Search the last ~3200 tweets from a given user.

        Parameters
            user_id - str
                The twitter user to search
            last_tweet_id - str
                The id of the user's last tweet
            tweets - List[str]
                A list of all tweets that have been fetched but not saved on
                disk yet.
            update_last_tweet - Dict[str, str]
                A dictionary mapping users ids to their last tweet id. This
                dictionary is reset when we save the data on disk.
            df_last - pd.core.frame.DataFrame
                A dataframe mapping user_id to last tweet id
        """
        print(f"Tweets fetched for user id {user_id} :")
        count = 1
        for tweet in Cursor(self.api.user_timeline,
                            id=user_id,
                            since_id=last_tweet_id
                            ).items():
            tweet_dict = tweet._json
            tweet_str = json.dumps(tweet._json)
            new_tweet_id = tweet_dict["id_str"]
            if int(new_tweet_id) > int(float(last_tweet_id)):
                last_tweet_id = new_tweet_id
            tweets.append(tweet_str)
            print(str(count), end='\r')
            #sys.stdout.write('\033[2K\033[1G')
            #sys.stdout.write(str(count))
            #sys.stdout.flush()
            count+=1
        print()
        update_last_tweet[user_id] = last_tweet_id

        # Write tweets on disk if there is enough data
        batch_size = self.config["raw_tweets_search_batch_size"]
        if len(tweets) >= batch_size:
            self._write_tweets(tweets)
            tweets.clear()
            # update the last_tweet_id column and save on disk
            for user_id, last in update_last_tweet.items():
                if user_id not in df_last.index:
                    raise KeyError(f"user_id '{user_id}' not in df_last.index")
                df_last.at[user_id, "last_tweet_id"] = last

            df_last.to_csv(self.config["sg_users_last_path"])
            update_last_tweet.clear()

    @accepts(Any)
    @returns(None)
    def search_users(self):
        """Search for twitter users that are known to be Swiss-German.

        It will fetch new tweets from known Swiss-German twitter users. This is
        done by reading a dataframe containing three columns :
            - user_id : the twitter user ID
            - last_tweet_id : the id of most recent tweet fetched from this user
              If no tweet have been fetched yet, we put last_tweet_id = 1
            - gsw_tweet_count : how many Swiss-German tweet have been found from
              this user.
        This method will store all the tweets as a txt file.
        """
        # Create the logging configuration
        path_log = os.path.join(self.config["dir_path_log"], "search.log")
        create_logging_config(path_log)

        # Read the current state of the Swiss-German users count file
        df_count = pd.read_csv(self.config["sg_users_count_path"],
                               index_col="user_id", dtype=str)
        df_count.index = df_count.index.astype(str)
        df_last = pd.read_csv(self.config["sg_users_last_path"],
                              index_col="user_id", dtype=str)
        df_last.index = df_last.index.astype(str)
        # In principle there should be no duplicates
        #df_last = df_last.loc[~df_last.index.duplicated(keep='first')]

        # Sometimes tweet_filter.py will find new Swiss-German users and store
        # them in the sg_users_count file. We need to get these new users and
        # include them in the sg_users_last file.
        self.update_sg_users_last(df_count, df_last)

        df_all = pd.merge(df_last, df_count, on="user_id")
        # 1st strategy : Separate the dataframe in two parts
        # This is because we want to put the priority on users that were never
        # searched, whatever the tweet count is. For users already fetched once,
        # we append them at the end of the sorted dataframe (descending order)
        # such that they will processed but only after the others.
        # 2nd strategy : Since an already fetched user will be fast to fetch
        # again (only a few new tweets from him) we can simply sort by gsw
        # tweet count. This is not the most effective if we restart the
        # streamer often.

        # 2nd strategy :
        df_all = df_all.sort_values(by="gsw_tweet_count", ascending=False)
        # 1st strategy :
        #df_new = df_all[df_all["last_tweet_id"]==1]
        #df_already_fetched_once = df_all[df_all["last_tweet_id"]!=1]
        #df_new = df_new.sort_values(by="gsw_tweet_count", ascending=False)
        #df_all = pd.concat([df_new, df_already_fetched_once])

        tweets = []
        # This will contains the last tweet id for each user from which we fetch
        # tweet. It will be used to update the last tweet id in the file.
        update_last_tweet = dict()

        for user_id, row in df_all.iterrows():
            try:
                self.search_user(str(user_id),
                                 str(row["last_tweet_id"]),
                                 tweets,
                                 update_last_tweet,
                                 df_last)
            except tweepy.error.TweepError as e:
                print(str(e))
                if "status code = 401" in str(e) or "status code = 404":
                    print("Cannot get user timeline (401/404), skipping...")
                    logging.exception("")
                else:
                    traceback.print_exc()
                    logging.exception("")
                    time.sleep(120)
                    self.authentify_twitter()
            except Exception:
                traceback.print_exc()
                logging.exception("")
                time.sleep(120)
                self.authentify_twitter()


        if len(tweets) > 0:
            self._write_tweets(tweets)
