from tweet_filter import *
from utils.utils import *
from torch import cuda
#import _thread
import time
import gc

def main():
    #_thread.start_new_thread( keep_alive, tuple() )
    base_time = 0
    config = load_yaml("config.yaml")
    cur_gsw_fetched = dict()
    cur_gsw_fetched["stream"] = 0
    cur_gsw_fetched["search"] = 0
    while True:
        current_time = time.time()
        if current_time > base_time + config["time_interval_process"]:
            base_time = current_time
            tweets = TweetFilter(config)
            tweets.process(cur_gsw_fetched)
            del(tweets.lid.model)
            del(tweets.lid.device)
            del(tweets.lid.config)
            del(tweets.lid.tokenizer)
            del(tweets.lid)
            del(tweets)
            gc.collect()
            cuda.empty_cache()
        time.sleep(10)

if __name__ == "__main__":
    main()
