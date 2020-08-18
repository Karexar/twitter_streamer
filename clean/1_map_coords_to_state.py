# Map coordinates to swiss state

import pandas as pd
from twitter.geocoder import *
from utils.utils import *
import os

###  Settings  #################################################################
dataset_path = "twitter/final_dataset/gsw_sentences.csv"
config_path = "twitter/config.yaml"
coords_to_state_path = "twitter/data/coords_to_state.pkl"
# Geocoding count interval used to save the mapping object regulary, to avoid
# having to query locationiq from the very beginning in case of error during the
# process
save_interval = 10
################################################################################

df = pd.read_csv(dataset_path)
#print(df.head())

config = load_yaml(config_path)
geocoder = Geocoder(config)

coords_str = set(df.coords.values)
print(str(len(coords_str)) + " coordinates to reverse geocode")

# Read the last saved file
coords_to_state = dict()
if os.path.exists(coords_to_state_path):
    coords_to_state = load_obj(coords_to_state_path)
print(str(len(coords_to_state)) + " coordinates already processed")

foreign_countries = set()

count = 0
for coord_str in list(coords_str):
    if not coord_str in coords_to_state:
        print("Reverse geocode " + str(coord_str))
        coord = coord_str.replace("(", "").replace(")", "").split(", ")
        lon = float(coord[0])
        lat = float(coord[1])
        country, state = geocoder.reverse_geocode_state(lon, lat)
        if country != "ch":
            foreign_countries.add(country)
            state = None
        print("  => " + str(state))
        coords_to_state[coord_str] = state
        time.sleep(1.1)
        count+=1
        if count > save_interval:
            print("Saving current object state...")
            save_obj(coords_to_state, coords_to_state_path)
            count=0

save_obj(coords_to_state, coords_to_state_path)
print("All coordinates reverse geocoded successfully")
