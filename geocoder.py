from utils.utils import *
import locationiq
from locationiq.rest import ApiException
import time
import re
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from typing import Tuple, Union, List
from typechecker.typecheck import *
from phrasal.norm_punc import normalize_text
import logging
from preprocessing.cleaner import *


class Geocoder:

    ch_states = sorted(["Thurgau","Glarus","Valais/Wallis","Basel-Landschaft",
                        "Solothurn","Bern","Zurich","Aargau","Basel-City",
                        "Obwalden","Schaffhausen","Nidwalden",
                        "Appenzell Innerrhoden","Neuchâtel","Ticino","Luzern",
                        "Schwyz","Vaud","Jura","Uri","Fribourg","Zug","Grisons",
                        "Appenzell Ausserrhoden","Sankt Gallen", "Geneva"])

    state_to_code = dict()
    state_to_code["Zurich"] = "ZH"
    state_to_code["Bern"] = "BE"
    state_to_code["Luzern"] = "LU"
    state_to_code["Aargau"] = "AG"
    state_to_code["Solothurn"] = "SO"
    state_to_code["Basel-City"] = "BS"
    state_to_code["Basel-stadt"] = "BS"
    state_to_code["Grisons"] = "GR"
    state_to_code["Zug"] = "ZG"
    state_to_code["Sankt Gallen"] = "SG"
    state_to_code["Basel-Landschaft"] = "BL"
    state_to_code["Thurgau"] = "TG"
    state_to_code["Valais/Wallis"] = "VS"
    state_to_code["Obwalden"] = "OW"
    state_to_code["Appenzell Ausserrhoden"] = "AR"
    state_to_code["Ticino"] = "TI"
    state_to_code["Appenzell Innerrhoden"] = "AI"
    state_to_code["Schwyz"] = "SZ"
    state_to_code["Nidwalden"] = "NW"
    state_to_code["Fribourg"] = "FR"
    state_to_code["Schaffhausen"] = "SH"
    state_to_code["Jura"] = "JU"
    state_to_code["Uri"] = "UR"
    state_to_code["Glarus"] = "GL"
    state_to_code["Vaud"] = "VD"
    state_to_code["Neuchâtel"] = "NE"
    state_to_code["Geneva"] = "GE"

    @accepts(Any, Union[str, dict])
    @returns(None)
    def __init__(self, config):
        """Prepare the geocoder and load the dictionary mapping
        location to coordinates"""
        self.config = load_yaml(config) if isinstance(config, str) else config
        credentials = load_yaml(self.config["credentials_path"])

        # Load the dict mapping address to a geolocalization object
        loc_to_coords_path = self.config["loc_to_coords_path"]
        if not os.path.exists(loc_to_coords_path):
            with open(loc_to_coords_path, "w", encoding="utf8"):
                pass
        self.loc_to_coords = load_dict_from_txt(loc_to_coords_path)
        # This will contain the file to append new mapping
        # This is stored as a text file, such that we can append easily each new
        # element without worrying about the python dict object not being
        # saved because of some Exception.
        self.loc_to_coords_file = None

        # Load the CH words (i.e. swiss city names, postal codes, and so on)
        self.ch_words = None
        with open(self.config["ch_words_path"], "r", encoding="utf8") as f:
            self.ch_words = f.readlines()
            self.ch_words = [x for x in self.ch_words if x != "\n" and x != ""]
            # Take everything before the first tab
            self.ch_words = [re.sub(r'\t[^\n]*\n', '', x) for x in self.ch_words]
            self.ch_words = [x.replace("\n", "") for x in self.ch_words]

        # Heavy normalize ch_words. We keep both list because we will need to
        # check if a normalized ch word is contained in the normalized location
        # field (user.location) of the twitter account. Then, we need the
        # original ch word to query the location with the api.
        self.ch_words_norm = [heavy_normalize_text(x) for x in self.ch_words]

        self.ch_polygon = Polygon([tuple(x) for x in self.config["ch_polygon"]])


        # Configure the locationiq API
        loc_config = locationiq.Configuration()
        loc_config.api_key["key"] = credentials["locationiq_api"]["key"]

        # Defining host is optional and default to https://eu1.locationiq.com/v1
        loc_config.host = self.config["locationiq"]["host"]

        # Enter a context with an instance of the API client
        with locationiq.ApiClient(loc_config) as api_client:
            self.api_instance = locationiq.SearchApi(api_client)
            self.api_instance_rev = locationiq.ReverseApi(api_client)

            self.gformat = self.config["locationiq"]["gformat"]
            self.normalizecity = self.config["locationiq"]["normalizecity"]
            self.addressdetails = self.config["locationiq"]["addressdetails"]
            self.viewbox = self.config["locationiq"]["viewbox"]
            self.bounded = self.config["locationiq"]["bounded"]
            self.limit = self.config["locationiq"]["limit"]
            self.accept_language = self.config["locationiq"]["accept_language"]
            self.countrycodes = self.config["locationiq"]["countrycodes"]
            self.namedetails = self.config["locationiq"]["namedetails"]
            self.dedupe = self.config["locationiq"]["dedupe"]
            self.extratags = self.config["locationiq"]["extratags"]
            self.statecode = self.config["locationiq"]["statecode"]
            self.matchquality = self.config["locationiq"]["matchquality"]
            self.postaladdress = self.config["locationiq"]["postaladdress"]


    @accepts(Any)
    @returns(None)
    def open_mapping_file(self):
        if self.loc_to_coords_file == None:
            f = open(self.config["loc_to_coords_path"], 'a+', encoding="utf8")
            self.loc_to_coords_file = f

    @accepts(Any)
    @returns(None)
    def clean(self):
        self.loc_to_coords_file.close()
        self.loc_to_coords_file = None

    @accepts(Any, List[Union[int, float]], List[List[Union[int, float]]])
    @returns(bool)
    def is_in_bounding_boxes(self, point, boxes):
        """Check if a point lies in one of the given bounding boxes, border
        included.

        Parameters
            point : List of [long, lat]
            boxes : List of [min_long, min_lat, max_long, max_lat]
        """

        for box in boxes:
            if point[0] >= box[0] \
            and point[0] <= box[2] \
            and point[1] >= box[1] \
            and point[1] <= box[3]:
                return True
        return False

    @accepts(Any, Tuple)
    @returns(bool)
    def are_coords_in_switzerland(self, coords: Tuple) -> bool:
        """Returns true if the coords [long, lat] are located in
        switzerland"""
        coords = Point(coords)
        return self.ch_polygon.contains(coords)

    @accepts(Any, str)
    @returns(Any)
    def get_ch_location(self, location_field):
        """From the user.location field, check if the location is known to be
        in Switzerland. This is done by checking for each CH word or word
        combination (city name, states...) if the word appear in the
        user.location field.
        """

        location_field_norm = heavy_normalize_text(location_field)

        # Check the city names and states
        for i in range(len(self.ch_words_norm)):
            ch_word_norm = self.ch_words_norm[i]
            if ch_word_norm in location_field_norm:
                # No alphanumeric character before or after the word
                # the preceding is to filter before using the regex, which is
                # much more time consuming than "ch_word in location_field_norm"
                if re.search(r"(^|\W)" + ch_word_norm
                             + r"(\W|$)", location_field_norm):
                    return self.ch_words[i]
        return None

    def locationiq_search(self, query):
        """Simply call the search function of the locationIQ api
        """
        loc = self.api_instance.search(query,
                                       self.gformat,
                                       self.normalizecity,
                                       addressdetails=self.addressdetails,
                                       viewbox=self.viewbox,
                                       bounded=self.bounded,
                                       limit=self.limit,
                                       accept_language=self.accept_language,
                                       countrycodes=self.countrycodes,
                                       namedetails=self.namedetails,
                                       dedupe=self.dedupe,
                                       extratags=self.extratags,
                                       statecode=self.statecode,
                                       matchquality=self.matchquality,
                                       postaladdress=self.postaladdress)
        return loc

    @accepts(Any, float, float)
    def reverse_geocode_state(self, lon, lat):
        """Takes GPS coordinates and returns the corresponding swiss state for
        each of these coordinate
        """

        try:
            api_response = self.api_instance_rev.reverse(lat, lon, "json", 1)
            return (api_response.address.country_code,
                    api_response.address.state)
        except locationiq.exceptions.ApiException as e:
            if e.reason == "Not Found":
                print(f"({lon},{lat}) not found)")
                return (None, None)
            raise
        except ApiException as e:
            raise


    @accepts(Any, str)
    @returns(Tuple[dict, str])
    def forward_geocode(self, query):
        """Forward geocode an address and store it in cache

        It will first check if the query is cached. If yes, return the address.
        If not, forward geocode the query. If a result is found, return. If not,
        check if there are CH words (e.g. a city name in switzerland) in the
        query. If not, return. If yes, forward geocode with this word only.

        Note : if location is for exemple "Freiburg, Germany", and locationiq is
        parametrised to find location in Switzerland only, then it will fallback
        into finding ch_words, which yields "Freiburg", and will responds
        incorrectly by "Freiburg, Switzerland". This should not be an issue
        since the text will be discarded later because not written in
        swiss-german. The alternative would be to parameterise locationiq to
        find location more globally, but in this case the swiss result may not
        appear in the results because of limitations in the response list size.

        query: str | address to geocode
        return: tuple | geographic information
        """
        # Remove special characters that should not appear in a location field
        query = normalize_text(query)
        query = Cleaner.remove_wrong_chars(query)
        query = re.sub(r"[^\w\s\.\,\-\\/\(\)\&\']", " ", query)
        query = re.sub(r"[/\r?\n|\r/]", " ", query)
        query = Cleaner.clean_spaces(query)
        query = query.lower().strip()
        if query in self.loc_to_coords:
            return self.loc_to_coords[query]
        elif len(query) < 2:
            loc_tuple = (dict(), "location not found")
            return loc_tuple
        else:
            try:
                self.open_mapping_file()
                msg = "Request to locationiq : '" + query + "'"
                print(msg)
                logging.info(msg)
                time.sleep(1.1) # locationiq limits 60 requests/minute
                loc = self.locationiq_search(query)
                loc = loc[0].to_dict()
                loc_type = "Geocoder_original"
                loc_tuple = (loc, loc_type)
                self.loc_to_coords[query] = loc_tuple
                loc_str = query + "\n" + str(loc_tuple) + "\n"
                self.loc_to_coords_file.write(loc_str)
                return loc_tuple
            except ApiException as e:
                if e.status == 404:
                    # Unable to find location, try to find CH words
                    ch_loc = self.get_ch_location(query)
                    if ch_loc is not None:
                        try:
                            print("Not found, request ch word instead : '"
                                  + str(ch_loc) + "'")
                            time.sleep(1) # locationiq limits 60 requests/minute
                            loc = self.locationiq_search(ch_loc)
                            loc = loc[0].to_dict()
                            loc_type = "Geocoder_CH_word"
                            loc_tuple = (loc, loc_type)
                            self.loc_to_coords[query] = loc_tuple
                            loc_str = query + "\n" + str(loc_tuple) + "\n"
                            self.loc_to_coords_file.write(loc_str)
                            return loc_tuple
                        except ApiException as e:
                            # Raise an exception if a ch word is found but no
                            # localization can be found. This should never
                            # happen (a city/state should always
                            # be geocoded successfully)
                            # update : not always true, the postal code list
                            # is approximative, some of the number included do
                            # not correspond to any city. So we simply print
                            # the output for manual check.
                            if e.status == 404:
                                error_str = "CH word localization not found\n" \
                                            + "Query: " + query + "\n" \
                                            + "CH word: " + ch_loc
                                print(error_str)
                                loc_tuple = (dict(), "location not found")
                                self.loc_to_coords[query] = loc_tuple
                                loc_str = query + "\n" + str(loc_tuple) + "\n"
                                self.loc_to_coords_file.write(loc_str)
                                return loc_tuple
                                #raise Exception(error_str)
                            else:
                                raise(e)
                    else:
                        loc_tuple = (dict(), "location not found")
                        self.loc_to_coords[query] = loc_tuple
                        loc_str = query + "\n" + str(loc_tuple) + "\n"
                        self.loc_to_coords_file.write(loc_str)
                        return loc_tuple
                else:
                    self.clean()
                    raise(e)

    @accepts(Any, str)
    @returns(Any)
    def postcode_to_state(self, postcode):
        if not (int(postcode) == -1 or
        (int(postcode) > 999 and int(postcode) < 10000)):
            raise ValueError(f"{postcode} is not a valid postcode")
        if postcode == "-1":
            return None
        print(f"Request to locationiq : '{postcode}'")
        time.sleep(1.1) # locationiq limits 60 requests/minute
        loc = self.locationiq_search(postcode)
        loc = loc[0].to_dict()
        state = loc["address"]["state"]
        return state

    @staticmethod
    @accepts(Any)
    @returns(Any)
    def get_state_code(state):
        """Convert a swiss state into state code (e.g. Zürich => ZH)
        The spelling of the state comes from the locationiq responses."""
        if state in Geocoder.state_to_code:
            return Geocoder.state_to_code[state]
        return None

    @staticmethod
    @accepts(Any)
    @returns(Any)
    def state_to_dialect(state):
        """Get a state (i.e. complete name of a canton), and returns the
        corresponding dialect"""

        code = Geocoder.get_state_code(state)
        if code in {"ZH", "AG"}:
            return "ZH"
        elif code in {"BS", "BL"}:
            return "NW"
        elif code in {"OW", "NW", "LU", "UR", "SZ", "ZG", "GL"}:
            return "CE"
        elif code in {"GR"}:
            return "GR"
        elif code in {"BE", "SO"}:
            return "BE"
        elif code in {"VS"}:
            return "VS"
        elif code in {"AR", "AI", "SG", "TG", "SH"}:
            return "EA"
        elif code in {"FR"}:
            return "RO"
        else:
            return None
