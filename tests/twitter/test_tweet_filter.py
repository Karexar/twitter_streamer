import pytest
from twitter.tweet_filter import *
import os
from utils.utils import *
from pytest import approx
from shutil import copyfile
import pandas as pd

test_config = load_yaml("tests/config.yaml")

def reset_processed_ids_file():
    """Reset the file containing all tweets ids already processed.
    This is done by copying the file 'processed_ids_src.txt' into
    'processed_ids'"""
    src = test_config["processed_tweets_ids_src_path"]
    dst = test_config["overwrite"]["processed_tweets_ids_path"]
    copyfile(src, dst)

@pytest.fixture(scope="module")
def tweets_obj():
    config = load_yaml(test_config["path_config"])
    for var_name in test_config["overwrite"]:
        if not var_name in config:
            raise ValueError("Config element to overwrite '" + var_name +
                             "' does not exist")
        config[var_name] = test_config["overwrite"][var_name]
    reset_processed_ids_file()

    tweet_filter = TweetFilter(config)
    # The 'tweets' attribute in initialize in the process method. To test
    # the other methods separately, we need to initialize 'tweets' here.
    # For now it will be the file contained in raw_tweets_dir_path for test.
    names = os.listdir(config["raw_tweets_stream_dir_path"])
    path = os.path.join(config["raw_tweets_stream_dir_path"], names[0])
    with open(path, "r", encoding="utf8") as f:
        raw_tweets = f.readlines()
        tweet_filter.tweets = [json.loads(x) for x in raw_tweets]

    return TweetFilter(config)

@pytest.fixture(scope="module")
def tweets_dict():
    tweets = dict()
    path_tweets = test_config["path_tweets"]
    for file_name in os.listdir(path_tweets):
        with open(os.path.join(path_tweets, file_name), "r",
                  encoding="utf8") as f:
            tweet = f.readline()
            tweet = json.loads(tweet)
            tweets[file_name[:-4]] = tweet
    return tweets

@pytest.fixture(scope="module")
def sentences_info():
    obj = [(2,
            "S Wappe zeigt links de Rhy, rächts s Sachseross, in "+
            "de Mitti di alti lippischi Rose uff silbrigem Grund.",
            (1.1, 2.2),
            "a"),
           (5,
            "Le lapin saute sur la table afin d'accéder aux " +
            " carottes qui traînaient depuis ce matin",
            (3.3,4.4),
            "b"),
           (7,
            "Good morning, I don't know what time it is, but I " +
            "am still very sleepy, maybe I will sleep a bit more",
            (5.5,6.6),
            "c"),
           (9,
            "Durch den systematischen Vergleich der miteinander " +
            "verwandten Nachfolgesprachen können Ursprachen bis zu " +
            "einem gewissen Grad erschlossen werden.",
            (7.7,8.8),
            "d"),
           (10,
            "S Wappe zeigt links de Rhy, rächts s Sachseross",
            (9.9,10.1),
            "e"),
           (11,
            "Le lapin saute sur la table afin d'accéder",
            (11.1, 12.2),
            "f"),
           (14,
            "Good morning, I don't know what time it is",
            (13.1, 14.1),
            "g"),
           (16,
            "Durch den systematischen Vergleich der miteinander",
            (15.1, 16.1),
            "h")]
    return obj

class Test_tweets:
    @pytest.mark.parametrize("key,expected",
                             [("tweet_with_gps", True),
                              ("tweet_with_place", True),
                              ("tweet_with_location", True),
                              ("tweet_no_geo", False)])
    def test_is_geo_available(self, tweets_dict, tweets_obj, key, expected):
        assert(tweets_obj.is_geo_available(tweets_dict[key]) == expected)

    @pytest.mark.parametrize("key,expected",
                             [("tweet_no_subtweet", 1),
                              ("tweet_with_quoted", 2),
                              ("tweet_with_retweeted", 1),
                              ("tweet_with_retweeted_quoted", 2)])
    def test_extract_sub_tweets(self, tweets_obj, tweets_dict, key, expected):
        sub_tweets = tweets_obj._extract_sub_tweets([tweets_dict[key]])
        assert(len(sub_tweets) == expected)

    def test_filter_out_duplicates(self, tweets_obj, tweets_dict):
        # Start with a no tweet processed
        tweets_obj.processed_tweets_ids = set()
        # Add a duplicate in the batch itself
        tweets = [tweets_dict["tweet_no_geo"]] * 2
        res = tweets_obj._filter_out_duplicates(tweets)
        assert(len(res) == 1)
        assert(res[0] == tweets_dict["tweet_no_geo"])
        # Rerun the method, this time the duplicate should be detected from
        # the processed_tweets_ids attribute
        res = tweets_obj._filter_out_duplicates(tweets)
        assert(len(res) == 0)

        # Mix of duplicate in batch and already processed
        tweets = [tweets_dict["tweet_no_geo"],
                  tweets_dict["tweet_no_subtweet"],
                  tweets_dict["tweet_with_gps"],
                  tweets_dict["tweet_with_gps"]]
        res = tweets_obj._filter_out_duplicates(tweets)
        assert(len(res) == 2)
        assert(res[0] == tweets_dict["tweet_no_subtweet"])
        assert(res[1] == tweets_dict["tweet_with_gps"])

    def test_filter_geo(self, tweets_obj, tweets_dict):
        tweets = [tweets_dict["tweet_with_gps"],
                  tweets_dict["tweet_with_place"],
                  tweets_dict["tweet_with_location"],
                  tweets_dict["tweet_no_geo"]]
        tweets_expected = tweets[:-1].copy()
        assert(tweets_obj._filter_geo(tweets) == tweets_expected)

    def test_extract_text_from_tweets(self, tweets_obj, tweets_dict):
        tweets = [tweets_dict["tweet_with_full_text"],
                  tweets_dict["tweet_with_extended"],
                  tweets_dict["tweet_with_text"]]
        expected = [(0, tweets[0]["full_text"]),
                    (1, tweets[1]["extended_tweet"]["full_text"]),
                    (2, tweets[2]["text"])]
        assert(tweets_obj._extract_text_from_tweets(tweets) == expected)
        with pytest.raises(ValueError,
                           match=r"^Cannot retrieve text from tweet$"):
            tweet = tweets_dict["tweet_with_full_text"]
            tweet.pop("full_text")
            tweets_obj._extract_text_from_tweets([tweet])

        tweet = tweets_dict["tweet_limit"]
        assert(tweets_obj._extract_text_from_tweets([tweet]) == [])

    @pytest.mark.parametrize("test_input,expected",
                 [([(1, "RT @bou-boule, so what ?"),
                    (2, "MT http://test.com best website ever @bob"),
                    (3, "Hey #greetings how are you ?")],
                   [(1, "so what ?"),
                    (2, "best website ever"),
                    (3, "Hey how are you ?")]),
                   ([(13, "Hey @bill check this out https://a.io")],
                    [(13, "Hey check this out")]),
                   ([], [])
                 ])
    def test_preprocess(self, test_input, expected, tweets_obj):
        assert(tweets_obj._preprocess(test_input) == expected)

    @pytest.mark.parametrize("test_input,expected",
                 [([(1, "test     test"), # 2 tabs
                    (2, "œil"),
                    (3, "his name is «Bill the great”")],
                   [(1, "test test"),
                    (2, "oeil"),
                    (3, "his name is \"Bill the great\"")]),
                   ([(13, "Hallo   Bill")], [(13, "Hallo Bill")]),
                   ([], [])
                 ])
    def test_normalize_text(self, test_input, expected, tweets_obj):
        assert(tweets_obj._normalize_texts(test_input) == expected)

    @pytest.mark.parametrize("test_input,expected",
        [([(1, "Hello ! How are you ? I would like to know how you did this !"),
           (2, "This is very sad. Nobody expected this.")],
          [(1, "Hello !"),
           (1, "How are you ?"),
           (1, "I would like to know how you did this !"),
           (2, "This is very sad."),
           (2, "Nobody expected this.")]),
          ([(1, "How are you ?")], [(1, "How are you ?")]),
          ([], [])
        ])
    def test_split_texts(self, test_input, expected, tweets_obj):
        assert(tweets_obj._split_texts(test_input) == expected)

    @pytest.mark.parametrize("test_input,expected",
        [([(1, "Ha ha ha, this is S H I T !"),
           (1, "Please, don't tell such stupid things"),
           (2, "LOL, no way, I DONT BELIEVE YOU"),
           (2, "How are you ?"),
           (3, "I would like to know how you did this !"),
           (3, "This is Elisabeth, Mark, Bill, and Lucy"),
           (3, "my num is 012 345 67 89")],
          [(1, "Please, don't tell such stupid things"),
           (3, "I would like to know how you did this !")]),
          ([(1, "How are you ?")], []),
          ([], [])
         ])
    def test_filter_valid_sentences(self, test_input, expected, tweets_obj):
        assert(tweets_obj._filter_valid_sentences(test_input) == expected)

    def test_geocode_tweets(self, tweets_obj):
        # self.tweets is created in the process funcion only, we need to
        # initialize it here
        with open(tweets_obj.raw_tweets_paths[0], "r", encoding="utf8") as f:
            raw_tweets = f.readlines()
            tweets_obj.tweets = [json.loads(x) for x in raw_tweets]
        idx_to_loc = tweets_obj._geocode_tweets([0])
        assert(idx_to_loc[0][2] == "from original location")
        assert(approx(idx_to_loc[0][1][0], 6.60014686413811))
        assert(approx(idx_to_loc[0][1][0], 46.6588912633543))

    def test_attach_gsw_location(self, tweets_obj):
        sentences = [(0, "a"), (1, "b"), (2, "c"), (3, "d")]
        idx_to_location = dict({0:((0,0), (10.287, 47.476), "a"),
                                1:((0,0), (8.904, 47.484), "b"), # in CH
                                2:((0,0), (7.652, 45.718), "c"),
                                3:((0,0), (6.573, 47.305), "d"),
                                4:((0,0), (16.573, 47.305), "e")})
        res = tweets_obj._attach_gsw_location(sentences, idx_to_location, False)
        assert(len(res) == 1 and res[0][0] == 1 and res[0][1] == "b"
               and approx(res[0][2][0], 8.904) and approx(res[0][2][0], 47.484)
               and res[0][3] == "b")
        res = tweets_obj._attach_gsw_location(sentences, idx_to_location, True)
        assert(len(res[0]) == 4)

    def test_filter_gsw_sentences(self, tweets_obj, sentences_info):
        res = tweets_obj._filter_gsw_sentences(sentences_info)
        assert(len(res) == 2)
        assert(len(res[0]) == 5)
        assert(len(res[1]) == 5)
        # Check the sentences
        assert(res[0][0] == sentences_info[0][1])
        assert(res[1][0] == sentences_info[4][1])
        # Check the coordinates
        assert(res[0][1] == sentences_info[0][2])
        assert(res[1][1] == sentences_info[4][2])
        # Check the prediction value
        threshold = tweets_obj.config["lid_threshold"]
        assert(res[0][2] >= threshold and res[0][2] <= 1.0)
        assert(res[1][2] >= threshold and res[1][2] <= 1.0)
        # Check the geo_source
        assert(res[0][3] == sentences_info[0][3])
        assert(res[1][3] == sentences_info[4][3])
        # Check the tweets
        assert(res[0][4] == tweets_obj.tweets[sentences_info[0][0]])
        assert(res[1][4] == tweets_obj.tweets[sentences_info[4][0]])

    def test_write_gsw_tweets(self, tweets_obj):
        dir_path = tweets_obj.config["out_dir_tweet_processing"]
        gsw_tweets = [("a,bc",(1.1,2.2), 0.97, "GPS", dict()),
                      ("def", (3.3,4.4), 0.93, "GPS", dict()),
                      ("gh,i", (5.5,6.6), 0.997, "GPS", dict())]
        files = os.listdir(dir_path)
        for file in files:
            os.remove(os.path.join(dir_path, file))
        for i in range(3):
            res = tweets_obj._write_gsw_tweets(gsw_tweets)
            files = sorted(os.listdir(dir_path))
            assert(len(files) == i+1)
            for j in range(i):
                print(i)
                print(j)
                for x in files:
                    print(x)
                assert(files[j] == str(j)+".pkl")
                path = os.path.join(dir_path, str(j)+".pkl")
                obj = load_obj(path)
                assert(len(obj) == 3)
                assert(obj[0] == gsw_tweets[0])
                assert(obj[1] == gsw_tweets[1])
                assert(obj[2] == gsw_tweets[2])
        files = os.listdir(dir_path)
        for file in files:
            os.remove(os.path.join(dir_path, file))

    def test_write_new_sg_users(self, tweets_obj):
        gsw_tweets = [("a,bc",(1.1,2.2), 0.998, "GPS", {"user":{"id_str":2}}),
                      ("def", (3.3,4.4), 0.93, "GPS", {"user":{"id_str":3}}),
                      ("gh,i", (5.5,6.6), 0.997, "GPS", {"user":{"id_str":4}})]
        # Make sure we start with an empty file
        df = pd.DataFrame([], columns=["user_id", "gsw_tweet_count"])
        df.set_index("user_id", inplace=True)
        df.to_csv(tweets_obj.config["sg_users_count_path"])
        tweets_obj._write_new_sg_users(gsw_tweets)
        df = pd.read_csv(tweets_obj.config["sg_users_count_path"])
        assert(df.shape == (2,2))
        assert(df.at[0, "user_id"] == 2)
        assert(df.at[1, "user_id"] == 4)
        assert(df.at[0, "gsw_tweet_count"] == 1)
        assert(df.at[1, "gsw_tweet_count"] == 1)
        # remove last line and save file
        df = df.drop(1)
        df.set_index("user_id", inplace=True)
        df.to_csv(tweets_obj.config["sg_users_count_path"])
        # write the same elements again
        tweets_obj._write_new_sg_users(gsw_tweets)
        df = pd.read_csv(tweets_obj.config["sg_users_count_path"])
        assert(df.shape == (2,2))
        assert(df.at[0, "user_id"] == 2)
        assert(df.at[1, "user_id"] == 4)
        assert(df.at[0, "gsw_tweet_count"] == 2)
        assert(df.at[1, "gsw_tweet_count"] == 1)

    def test_update_processed_tweets(self, tweets_obj):
        tweets_obj.new_tweets_ids = {1,13,15}
        tweets_obj._update_processed_tweets()
        path = tweets_obj.config["processed_tweets_ids_path"]
        with open(path, "r", encoding="utf8") as f:
            lines = f.readlines()
            assert(lines[0] == "11\n")
            assert(lines[1] == "12\n")
            assert(lines[2] == "1\n")
            assert(lines[3] == "13\n")
            assert(lines[4] == "15\n")
