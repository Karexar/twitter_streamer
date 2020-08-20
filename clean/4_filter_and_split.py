


###  Settings  #################################################################
config_path = "config.yaml"
dataset_path = "dirty_dataset/gsw_tweets.pkl"
useless_locs_path = "data/useless_locs.pkl"
useful_locs_path = "data/useful_locs.pkl"
label_corrections_path = "data/label_corrections.pkl"
output_dir = "final_dataset"

# this dataset will be used to test what is the best method to label users
# for the self learning part. We will treat the training set as if it was
# unlabelled, then predict the dialect for each user, then compare to the
# real labels to see which user-labelling method works the best.
gsw_fake_unlabelled_dir = "data/twitter/fake_unlabelled"
# the dominant threshold is the minimum proportion a dialect should have for a
# given user to consider that the user is representative of the dialect.
# This is relevant only when the location is the tweet location (geo_source =
# GPS or Twitter_place) because each tweet may have a different location.
dominant_threshold = 0.67
################################################################################

# Split the dataset into a labelled and unlabelled set
df_unlab = df[df["dialect"] == "Unknown"]
df_lab = df[df["dialect"] != "Unknown"]
if df_unlab.shape[0] + df_lab.shape[0] != df.shape[0]:
    raise Exception(f"Wrong shapes : {df_unlab.shape[0]} + {df_lab.shape[0]} " +
                    f" != {df.shape[0]}")
# drop Fribourg
df_lab = df_lab[df_lab["label"]!="RO"]

# Save the fake unlabelled dataset
path = os.path.join(gsw_fake_unlabelled_dir, "full.csv")
df_lab.to_csv(path, sep="\t", header=False, index=False)

df_lab = df_lab.drop(columns=["user_id"])
print("***********************************************************************")
print("Final dataset")
print(f"Total length : {df.shape[0]}")
print(f"Length labelled : {df_lab.shape[0]}")
print(f"Length unlabelled : {df_unlab.shape[0]}")
print("***********************************************************************")

# Save on disk
path = os.path.join(gsw_labelled_dir, "full.csv")
df_lab.to_csv(path, sep="\t", header=False, index=False)
path = os.path.join(gsw_unlabelled_dir, "full.csv")
df_unlab.to_csv(path, sep="\t", header=False, index=False)
