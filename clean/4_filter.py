# This script filter the final dataset to remove sentences that are likely not
# to be Swiss-German, either by using the prediction score, or by removing
# users with low GSW tweet count.

import pandas as pd

###  Settings  #################################################################
dataset_path = "final_dataset/gsw_sentences.csv"
output_path = "final_dataset/gsw_filtered.csv"
gsw_threshold = 0.95
gsw_high_threshold = 0.995
gsw_count_threshold_labelled = 3
gsw_count_threshold_unlabelled = 10
################################################################################

print("Loading dataset")
df = pd.read_csv(dataset_path)

print("Filtering...")
# filter low gsw prediction
df = df[df.prediction >= gsw_threshold]

# For each user, compute the count of sentences of gsw prediction over a
# threshold.
user_to_gsw_count = df.groupby("user_id")["prediction"].apply(lambda x:
                        len([y for y in x if y > gsw_high_threshold]))
df = df.merge(user_to_gsw_count, left_on="user_id", right_on="user_id")
df = df.rename(columns={"prediction_x": "prediction"})
df = df.rename(columns={"prediction_y": "user_gsw_count"})

# Split into a labelled and unlabelled dataset
df_unlab = df[df["dialect"] == "Unknown"]
df_lab = df[df["dialect"] != "Unknown"]

# For the labelled dataset, the threshold is lower because since
# we know users are coming from Switzerland, the probability they are writing
# Swiss-German is higher
print(f"Labelled before : {df_lab.shape[0]}")
df_lab = df_lab[df_lab.user_gsw_count >= gsw_count_threshold_labelled]
print(f"Labelled after : {df_lab.shape[0]}")
print(f"Unlabelled before : {df_unlab.shape[0]}")
df_unlab = df_unlab[df_unlab.user_gsw_count >= gsw_count_threshold_unlabelled]
print(f"Unlabelled after : {df_unlab.shape[0]}")

df = pd.concat([df_lab, df_unlab])
df = df.drop(columns=["user_gsw_count", "location"])

print(f"Total length : {df.shape[0]}")

# Save on disk
df.to_csv(output_path, sep="\t", index=False)
