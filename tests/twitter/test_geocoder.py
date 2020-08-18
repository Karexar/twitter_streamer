import pytest
from twitter.geocoder import *
from shutil import copyfile
#import os
#from utils.utils import *

test_config = load_yaml("tests/config.yaml")

def reset_loc_to_coords_file():
    """Reset the file mapping the queries to locations. This is done by copying
    the file 'loc_to_coords_src.txt' into 'loc_to_coords'"""
    src = test_config["loc_to_coords_src_path"]
    dst = test_config["overwrite"]["loc_to_coords_path"]
    copyfile(src, dst)

@pytest.fixture(scope="module")
def geocoder():
    reset_loc_to_coords_file()
    config = load_yaml(test_config["path_config"])
    for var_name in test_config["overwrite"]:
        if not var_name in config:
            raise ValueError("Config element to overwrite '" + var_name +
                             "' does not exist")
        config[var_name] = test_config["overwrite"][var_name]
    return Geocoder(config)

@pytest.mark.parametrize("point, boxes, expected",
                         [([1,1],[[0,0,2,2],[2,0,3,1]], True),
                          ([0,2],[[0,0,2,2],[2,0,3,1]], True),
                          ([1,3],[[0,0,2,2],[2,0,3,1]], False),
                          ([4,1],[[0,0,2,2],[2,0,3,1]], False),
                          ([-1,1],[[0,0,2,2],[2,0,3,1]], False),
                          ([1,-1],[[0,0,2,2],[2,0,3,1]], False),
                          ([2.5,0.5],[[0,0,2,2],[2,0,3,1]], True),
                          ([2.5,1.5],[[0,0,2,2],[2,0,3,1]], False)])
def test_is_in_bounding_boxes(geocoder, point, boxes, expected):
    assert(geocoder.is_in_bounding_boxes(point, boxes) == expected)

@pytest.mark.parametrize("text, expected",
    [("I LIVE in >>>Goûmoens'la-ville<<< !!!", "Goumoëns"),
     ("1376, with my cows", "1376"),
     ("6.1376, 47.1376", None)])

def test_get_ch_location(geocoder, text, expected):
    assert(geocoder.get_ch_location(text) == expected)

def test_forward_geocode(geocoder):
    query = "1376"
    assert(query in geocoder.loc_to_coords)
    res = geocoder.forward_geocode(query)
    assert(res[0]["lat"] == "46.6588912633543")
    assert(res[0]["lon"] == "6.60014686413811")
    assert(res[1] == "from original location")
    query = "Echallens"
    assert(not query in geocoder.loc_to_coords)
    res = geocoder.forward_geocode(query)
    assert(query.lower() in geocoder.loc_to_coords)
    geocoder.clean() # close the loc_to_coords.txt file
    loc = load_dict_from_txt(test_config["overwrite"]["loc_to_coords_path"])
    assert(query.lower() in loc)
