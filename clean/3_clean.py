# This script cleans the twitter dataset
# WARNING : The check_location script must have been performed before, in order
# to have the useful and useless locations stored.

# This file loads the twitter dataset and performs the following operations :
#  1. Load the coords_to_state mapping object and create a column with the
#     cantons (states) from the coords column.
#  2. Map the coords to dialect for the GPS dataset (i.e. the data points where
#     coordinates are given by twitter).We take the dominant dialect for each
#     user.
#  3. Map the coord to canton for the IQ dataset (i.e. the data points where
#     coordinates are given by locationIQ).
#  4. Load the useful and useless locations, ensure all locations in the dataset
#     are included in one of the two files, and label (i.e. overwrite) as
#     "Unknown" all locations that are useless (only for the iq dataset)
#  5. Map each canton to the dialect (group of cantons) for the IQ dataset
#  6. Keep a 'sentence', 'label', and 'user_id' column and merge the two
#     datasets
#  7. Split the dataset into labelled and unlabelled, save them on disk and
#     output aggregated data

import pandas as pd
#import _thread
from utils.utils import *
from pathlib import Path
from twitter.geocoder import *
import os


###  Settings  #################################################################
config_path = "config.yaml"
dataset_path = "dirty_dataset/gsw_tweets.pkl"
useless_locs_path = "data/useless_locs.pkl"
useful_locs_path = "data/useful_locs.pkl"
label_corrections_path = "data/label_corrections.pkl"
output_dir = "final_dataset"
# the dominant threshold is the minimum proportion a dialect should have for a
# given user to consider that the user is representative of the dialect.
# This is relevant only when the location is the tweet location (geo_source =
# GPS or Twitter_place) because each tweet may have a different location.
dominant_threshold = 0.67
################################################################################

def dominant_dialect(dialects):
    """Return the dominant dialect or None if the proportion is not enough"""
    length = len(dialects)
    distribution = dialects.value_counts()
    if len(distribution) > 0:
        dominant_percentage = distribution[0] / length
        if dominant_percentage >= dominant_threshold:
            return distribution.index[0]
    return None


#_thread.start_new_thread( keep_alive, tuple() )

Path(gsw_labelled_dir).mkdir(parents=True, exist_ok=True)
Path(gsw_unlabelled_dir).mkdir(parents=True, exist_ok=True)
Path(gsw_fake_unlabelled_dir).mkdir(parents=True, exist_ok=True)

print("Loading the dataset...")
data = load_obj(dataset_path)


# 0: sentence, 1: (lon,lat), 2: gsw prediction, 3: geo_source, 4: original tweet
lines = [[x[0], str(x[1]), x[2], x[3], x[4]["user"]["location"],
         x[4]["user"]["id_str"]] for x in data]
df = pd.DataFrame(lines, columns=["sentence",
                                  "coords",
                                  "prediction",
                                  "geo_source",
                                  "location",
                                  "user_id"])
df["location"] = df["location"].map(lambda x: "" if x is None else str(x))
df["location"] = df["location"].map(heavy_normalize_text)


# Load the mapping from coords to canton (state)
coords_to_state_path = "twitter/data/coords_to_state.pkl"
coords_to_state = load_obj(coords_to_state_path)
if not set(df.coords.values).issubset(coords_to_state.keys()):
    raise Exception("coords_to_state must contain all coords in the dataset")

# Separate the dataset in two parts :
#  - first with coords directly available (geosource = GPS or twitter_place)
#  - second with geocoded coordinates
# Note that loc_type=="" means either that the location was not found by
# locationIQ, or that the user.location field was not given. In any case,
# we don't take these sentences for now because there is no geographic info.
twitter_coords_mask = ((df["geo_source"] == "GPS") |
                       (df["geo_source"] == "Twitter_place"))
df_gps = df[twitter_coords_mask].copy()
df_iq = df[~twitter_coords_mask].copy()
df_with_geo_size = df[df.geo_source!=""].shape[0]
if df.shape[0] != df_gps.shape[0] + df_iq.shape[0]:
    raise Exception(f"Wrong shapes ({df_gps.shape[0]}+{df_iq.shape[0]} != "
                    + f"{df.shape[0]})")

### 'GPS' dataset, where coordinates are given by twitter ######################
# for the first dataset, map the coords to the corresponding canton
print("-----------------------------------------------------------------------")
print("Processing the dataset with coordinates from twitter...")
print(f"Length : {df_gps.shape[0]}")
df_gps["canton"] = df_gps["coords"].map(lambda x: coords_to_state[x])
# Drop when canton = None. This should never happen unless the user in in a
# country with no state (e.g. Luxembourg).
#df_gps = df_gps.dropna(subset=["canton"])
df_gps["dialect"] = df_gps["canton"].map(Geocoder.state_to_dialect)
print("*****  Make sure the following are not swiss-german canton *****")
# If some swiss-german canton are printed, then we need to add them to the
# state_to_code dictionary
print(set(df_gps[df_gps["dialect"].isna()]["canton"]))

# Set the dominant dialect used for each user
user_dialect = df_gps.groupby("user_id")["dialect"].apply(dominant_dialect)
df_gps = df_gps.drop(["dialect"], axis=1)
# Merge the dominant dialect with the gps dataset
df_gps = df_gps.merge(user_dialect, on="user_id")
print("df_gps done. Summary : ")
value_count = df_gps["dialect"].value_counts()
print(value_count)
sum_labelled = value_count.sum()
print(f"Length labelled : {sum_labelled}")
sum_unlabelled = df_gps["dialect"].isna().sum()
print(f"Length unlabelled : {sum_unlabelled}")
if sum_labelled + sum_unlabelled != df_gps.shape[0]:
    raise Exception(f"Wrong shapes : {sum_labelled} + {sum_unlabelled} != " +
                    f"{df_gps.shape[0]}")

### 'IQ' dataset, where coordinates are given by locationIQ ####################
# map the coords to the corresponding canton
print("-----------------------------------------------------------------------")
print("Processing the dataset with coordinates from locationIQ...")
print(f"Length : {df_iq.shape[0]}")
df_iq["canton"] = df_iq["coords"].map(lambda x: coords_to_state[x])
# Overwrite the 'canton' with 'unknown' where location is useless
useless_locs = load_obj(useless_locs_path)
useful_locs = load_obj(useful_locs_path)
usefulness_set = useless_locs.union(useful_locs)

df_iq["useful"] = df_iq["location"].map(lambda x: 1 if x in useful_locs else 0)
print("Useful count : " + str(df_iq[df_iq["useful"]==1].shape[0]))
print("Useless count : " + str(df_iq[df_iq["useful"]==0].shape[0]))
df_iq.loc[df_iq["useful"]==0, "canton"] = "Unknown"

# Load the file containing the label corrections
corr = load_obj(label_corrections_path)
# Correct the label for the given locations
for key in corr:
    df_iq.loc[df_iq["location"]==key, "canton"] = corr[key]

# Map the canton to the dialect
df_iq["dialect"] = df_iq["canton"].map(Geocoder.state_to_dialect)
print("Make sure the following are not swiss-german canton : ")
# If some swiss-german canton are printed, then we need to add them to the
# state_to_code dictionary. It is hard to avoid this check because locationIQ
# gives the location in different language sometimes.
print(set(df_iq[df_iq["dialect"].isna()]["canton"]))

print("df_iq done. Summary : ")
value_count = df_iq["dialect"].value_counts()
print(value_count)
sum_labelled = value_count.sum()
print(f"Length labelled : {sum_labelled}")
sum_unlabelled = df_iq["dialect"].isna().sum()
print(f"Length unlabelled : {sum_unlabelled}")
if sum_labelled + sum_unlabelled != df_iq.shape[0]:
    raise Exception(f"Wrong shapes : {sum_labelled} + {sum_unlabelled} != " +
                    f"{df_iq.shape[0]}")

df_iq = df_iq.drop(["useful"], axis=1)

# Merge the dataset
df = pd.concat([df_gps, df_iq])

df[df["dialect"].isna()]["dialect"] = "Unknown"

if df.shape[0] != len(lines):
    raise Exception("The output is not the same size as the input")

df.to_csv("")
