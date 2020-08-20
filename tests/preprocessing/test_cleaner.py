from preprocessing.cleaner import *
import pytest

twitter_regexs = [
    (r'^RT\s', ''),
    (r'^MT\s', ''),
    (r'@\S*($|\s)', ' '),
    (r'#\S*($|\s)', ' '),
    (r'https?[\w\.\:\/]*($|\s)', ' '),
]

@pytest.mark.parametrize("sentence, regexs, expected", [
                             ("abc", [], "abc"),
                             ("abc", [(r"b", "d")], "adc"),
                             ("RT @jules This is nonsense, check this : " +
                              "https://www.test.com",
                              twitter_regexs,
                              " This is nonsense, check this :  "),
                             ("MT so cool !!! #happy yes",
                              twitter_regexs,
                              "so cool !!!  yes")
                         ])
def test_preprocess(sentence, regexs, expected):
    assert(Cleaner.preprocess(sentence, regexs) == expected)


# @pytest.mark.parametrize("sentence, expected",
#                          [("Hello++ †Ω°, my name is **LUC** _yeah_",
#                            "Hello      , my name is   LUC    yeah ")])
# def test_remove_special_chars(sentence, expected):
#     assert(Cleaner.remove_special_chars(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [(" Hello      , my name is   LUC    yeah .  ",
                           "Hello , my name is LUC yeah .")])
def test_clean_spaces(sentence, expected):
    assert(Cleaner.clean_spaces(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("hello :) good ^^ -.- or good :-))))))",
                             "hello   good     or good  ")
                         ])
def test_remove_smileys(sentence, expected):
    assert(Cleaner.remove_smileys(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("hello ^gf", "hello ")
                         ])
def test_remove_hat_element(sentence, expected):
    assert(Cleaner.remove_hat_element(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("hey &gt bye", "hey   bye"),
                            ("hey &le; bye", "hey   bye"),
                         ])
def test_remove_html_entities(sentence, expected):
    assert(Cleaner.remove_html_entities(sentence) == expected)


@pytest.mark.parametrize("sentence, expected, from_size",
                         [
                            ("hey && how - are you%^ (fine): bye",
                             "hey how are you%^ (fine): bye", 1),
                            ("hey && how - are you%^ (fine): bye",
                             "hey how - are you%^ (fine): bye", 2),
                            ("hey && how - are you%^ (fine): bye",
                             "hey && how - are you%^ (fine): bye", 3)
                         ])
def test_remove_groups_of_special_chars(sentence, expected, from_size):
    assert(Cleaner.remove_groups_of_special_chars(sentence,
                                                  from_size)==expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("hey ² gt \\°² bye \\", "hey gt \\°² bye"),
                         ])
def test_remove_isolated_special_chars(sentence, expected):
    assert(Cleaner.remove_isolated_special_chars(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("hey | (how are you) { g",
                            ["hey", "how are you", "g"]),
                         ])
def test_split_parenthesis(sentence, expected):
    assert(Cleaner.split_parenthesis(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("hey ¦ goo§d", "hey   goo d")
                         ])
def test_remove_not_good_chars(sentence, expected):
    good_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    good_chars += "ÀÁÂÄÈÉÊËÍÌÎÏÓÒÔÖÚÙÛÜàáâäèéêëìíîïôöòóüùúûÿ"
    good_chars += " -,.?!0123456789%&\"\'()/$*+:;<=>[]\\^_{}|\\~€°²"
    good_chars = set(good_chars)
    assert(Cleaner.remove_not_good_chars(sentence, good_chars) == expected)


@pytest.mark.parametrize("sentence, expected, special",
                         [
                            ("hey ¦ how are @ #you",
                             "hey ¦ how are    you",
                             set("#@[]{}<>=^\\_~")),
                            ("hey ¦ how are @ #you",
                             "hey   how are    you",
                             set("¦#@[]{}<>=^\\_~"))
                         ])
def test_remove_special_chars(sentence, expected, special):
    assert(Cleaner.remove_special_chars(sentence, special_chars=special)
           == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("hey ¦ goo§d bye$", "hey bye$")
                         ])
def test_remove_special_words(sentence, expected):
    assert(Cleaner.remove_special_words(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("hey!!!How are you ??! Fine,.and you.. he ;go....",
                             "hey! How are you? Fine. and you... he; go...")
                         ])
def test_clean_punc(sentence, expected):
    assert(Cleaner.clean_punc(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("hey   How are you  ! Fine and  you ",
                             "hey How are you ! Fine and you")
                         ])
def test_clean_spaces(sentence, expected):
    assert(Cleaner.clean_spaces(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("I have 9'000 cats", "I have  <num>  cats"),
                            ("big 1000.00'..,00 number", "big  <num>  number")
                         ])
def test_add_num_token(sentence, expected):
    assert(Cleaner.add_num_token(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("Hi! how are you ? hmm; good... and you, good?!",
                             "Hi <punc>  how are you  <punc>  hmm <punc>  " +
                             "good <punc>  and you <punc>  good <punc> ")
                         ])
def test_add_punc_token(sentence, expected):
    assert(Cleaner.add_punc_token(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("I have +++10 yep )))", "I have +10 yep )"),
                         ])
def test_remove_special_duplication(sentence, expected):
    assert(Cleaner.remove_special_duplication(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [
                            ("Salut Léon, à table äh",
                             "Salut Leon, a table äh")
                         ])
def test_remove_non_gsw_accent(sentence, expected):
    assert(Cleaner.remove_non_gsw_accent(sentence) == expected)
