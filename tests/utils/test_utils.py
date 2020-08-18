import pytest
from utils.utils import *

@pytest.mark.parametrize("text, expected",
    [("I LIVE in >>>Goumoëns'la-ville<<< !!!", "i live in goumoens la ville"),
     ("Well..... I am HERE !!!", "well i am here"),
     ("@Bill O'clash /!\\#thug/!\\", "bill o clash thug")])
def test_heavy_normalize_text(text, expected):
    assert(heavy_normalize_text(text) == expected)
