from streamer import *
#from utils.utils import *
#import _thread

def main():
    #_thread.start_new_thread( keep_alive, tuple() )
    gsw_stream = GSW_stream("config.yaml")
    gsw_stream.stream()

if __name__ == "__main__":
    main()
