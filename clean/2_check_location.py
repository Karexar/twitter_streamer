# This script loads the twitter dataset, heavy normalize the location field,
# and ask for each location if the location is valid or not. Valid means the
# location is specific enough at least at the canton (state) level (i.e.
# "Swizerland" is not valid because not specific enough). A location is not
# valid if several canton are included in the location (e.g. "Bern, Zürich").
# If a location contains two cantons and we are sure these two cantons would
# have the same label (e.g. "Bern, Solothurn" => "BE" for the pmk dataset), then
# the location can be considered as valid. All invalid location means the
# corresponding sentence will be labelled as unknown and will be a candidate
# for semi-supervised learning.

# This script also shows the inferred canton (state) from the location field.
# In case the inferred canton is not correct, but the location is valid, the
# user can rename the label

# This filtering part is necessary because locationIQ is very permissive. When
# it receives a query to locate in Switzerland, it will often find something,
# even if it's only very loosely related. Improvement can be made for example
# by checking the confidence score in the locationiq response, but it will still
# be an approximation.

# Automate this process is very hard because there are lots of corner cases.
# If we have access to an exhaustive list of all cities, state, districts in
# Switzerland, we may geocode only the location where such an element appears in
# the location, but on the first hand it's hard to be exhaustive (people may
# give another kind of location, e.g. mountain, lake, or river name) and on the
# other hand, some city names also exists elsewhere (Freiburg in Germany), or
# exists in several cantons in Switzerland (Aesch BL/ZU/LU). There is also the
# case where a city name may be interpreted differently by locationIQ (e.g.
# 'Rue' is a city in Fribourg, but also mean 'street' so can lead to any
# location response. Also 'Port' is a municipality in Bern, but may also mean
# 'harbor', which could lead to lots of places). All this corner cases led to
# the choice of performing a manual check on each location to check the
# validity.

# command during process :
#  - y : the location is valid
#  - n : the location is not valid
#  - s : save the current state
#  - q : quit
#  - r : like y but also correct the label name

import pandas as pd
from utils.utils import *
import os
from tqdm import tqdm
#import _thread

###  Settings  #################################################################
dataset_path = "dirty_dataset/gsw_tweets.pkl"
useless_locs_path = "data/useless_locs.pkl"
useful_locs_path = "data/useful_locs.pkl"
coords_to_state_path = "data/coords_to_state.pkl"
label_corrections_path = "data/label_corrections.pkl"
################################################################################

#_thread.start_new_thread( keep_alive, tuple() )

print("Loading the dataset...")
data = load_obj(dataset_path)
print("Total sentence count : " + str(len(data)))

print("Loading mapping coords to location...")
coords_to_state = load_obj(coords_to_state_path)
ch_states = sorted(["Thurgau","Glarus","Valais/Wallis","Basel-Landschaft",
                    "Solothurn","Bern","Zurich","Aargau","Basel-City",
                    "Obwalden","Schaffhausen","Nidwalden",
                    "Appenzell Innerrhoden","Neuchâtel","Ticino","Luzern",
                    "Schwyz","Vaud","Jura","Uri","Fribourg","Zug","Grisons",
                    "Appenzell Ausserrhoden","Sankt Gallen", "Geneva"])

print("Extracting locations...")
locs = [[str(x[1]), x[5]["user"]["location"]] for x in data
         if x[3] != "GPS" and x[3] != "Twitter_place"]
print("Sentences with location geocoded : " + str(len(locs)))
df = pd.DataFrame(locs, columns=["coords", "location"])
# Drop here to speed up the mapping next line
df = df.drop_duplicates(subset="location").dropna()
df["location"] = df["location"].map(heavy_normalize_text)
# Drop also here to remove new duplicates after normalization
df = df.drop_duplicates(subset="location").dropna()
df["state"] = df["coords"].map(lambda x: coords_to_state[x])
df = df.dropna(subset=["state"])
print("Unique locations count : " + str(df.shape[0]))

# Load the list of useless location. This corresponds to locations that were
# found to be in switzerland by the locationIQ api, but that are in fact not
# specific enough (e.g. "Switzerland", "CH") or simply wrong (e.g. "42",
# "earth")
print("Loading already processed locations...")
useless_locs = set()
useful_locs = set()
if os.path.exists(useless_locs_path):
    useless_locs = load_obj(useless_locs_path)
if os.path.exists(useful_locs_path):
    useful_locs = load_obj(useful_locs_path)

# Load the corrected mapping from location to correct canton
label_corrections = dict()
if os.path.exists(label_corrections_path):
    label_corrections = load_obj(label_corrections_path)

print("Useless locations : " + str(len(useless_locs)))
print("Useful locations : " + str(len(useful_locs)))

# check for each remaining location if the location is valid. A location is
# valid if it is specific enough to infer the state in Switzerland.
quit = False
count = 0
for _, row in df.iterrows():
    count += 1
    location = row[1]
    state = row[2]
    if not location in useless_locs and not location in useful_locs:
        progress = "(" + str(count) + "/" + str(df.shape[0]) + ")"
        keep_going = True
        while keep_going and not quit:
            if location is not None and state is not None:
                action = input("\n" + progress + " '" + location + "' => '"
                               + state + "' : ")
                if action == "y":
                    useful_locs.add(location)
                    keep_going = False
                #elif action == "s":
                #    save_obj(useless_locs, useless_locs_path)
                #    save_obj(useful_locs, useful_locs_path)
                elif action == "n":
                    useless_locs.add(location)
                    keep_going = False
                elif action == "q":
                    quit = True
                    break
                elif action == "r":
                    print("Locations available : ")
                    for i in range(len(ch_states)):
                        print(f"  {i+1}. {ch_states[i]}")
                    loop_new_loc = True
                    while loop_new_loc:
                        new_loc = input("Choose index of new location : ")
                        if (not new_loc.isnumeric()
                            or int(new_loc) < 1
                            or int(new_loc) > 26):
                            print("Error : choose a number between 1 and 26")
                        else:
                            loop_new_loc = False
                            keep_going = False
                            useful_locs.add(location)
                            label_corrections[location] = ch_states[int(new_loc)-1]
                            save_obj(label_corrections, label_corrections_path)
                save_obj(useless_locs, useless_locs_path)
                save_obj(useful_locs, useful_locs_path)
            else:
                print("Location or state not available")
                print("Location : " + str(location))
                print("State : " +str(state))
                useless_locs.add(location)
                keep_going = False
    if quit:
        break

save_obj(useless_locs, useless_locs_path)
save_obj(useful_locs, useful_locs_path)
