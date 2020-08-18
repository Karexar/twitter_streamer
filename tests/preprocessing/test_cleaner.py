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


@pytest.mark.parametrize("sentence, expected",
                         [("Hello++ †Ω°, my name is **LUC** _yeah_",
                           "Hello      , my name is   LUC    yeah ")])
def test_remove_special_chars(sentence, expected):
    assert(Cleaner.remove_special_chars(sentence) == expected)


@pytest.mark.parametrize("sentence, expected",
                         [("Hello      , my name is   LUC    yeah .",
                           "Hello, my name is LUC yeah.")])
def test_clean_spaces(sentence, expected):
    assert(Cleaner.clean_spaces(sentence) == expected)
