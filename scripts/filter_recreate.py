# This scripts takes all the raw tweets saved until know and perform
# the filter process once again on all these tweets. This is used
# if at some point we want a different processing for the tweet
# (cleaning, filtering...)

from tweet_filter_recreate import *
#from utils.utils import *
#import _thread

def main():
    #_thread.start_new_thread( keep_alive, tuple() )
    base_time = 0
    config = load_yaml("config.yaml")
    cur_gsw_fetched = dict()
    cur_gsw_fetched["stream"] = 0
    cur_gsw_fetched["search"] = 0
    tweets = TweetFilterRecreate(config)
    tweets.process(cur_gsw_fetched, config["path_dirty_gsw_tweets"])


if __name__ == "__main__":
    main()
