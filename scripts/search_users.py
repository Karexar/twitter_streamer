from streamer import *
from utils.utils import *
#import _thread
import sys

def main():
    """Call GSW_stream.search_users periodically, to fetch tweets from known
    Swiss-German twitter users
    """

    #_thread.start_new_thread( keep_alive, tuple() )
    base_time = 0
    config = load_yaml("config.yaml")
    while True:
        current_time = time.time()
        if current_time > base_time + config["time_interval_search_users"]:
            base_time = current_time
            streamer = GSW_stream("config.yaml")
            streamer.search_users()
        time.sleep(10)

if __name__ == "__main__":
    main()
