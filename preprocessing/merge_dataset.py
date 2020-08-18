import pandas as pd
import os

###  Settings  ################################################################
data_dirs = ["data/pmk/clean/",
             "data/twitter/clean",
             "data/whatsup/labelled"]
out_dir = "data/mix3/"
###############################################################################

df_train = pd.DataFrame()
df_test = pd.DataFrame()
for data_dir in data_dirs:
    # train set
    path = os.path.join(data_dir, "train.csv")
    print(f"Reading {path}")
    df = pd.read_csv(path, header=None, sep="\t")
    print(f"Length : {df.shape[0]}")
    path_dev = os.path.join(data_dir, "dev.csv")
    if os.path.exists(path_dev):
        print(f"Reading {path_dev}")
        df_dev = pd.read_csv(path_dev, header=None, sep="\t")
        print(f"Length : {df.shape[0]}")
        df = pd.concat([df, df_dev])
        print(f"Length train+dev : {df.shape[0]}")
    df_train = pd.concat([df_train, df])

    # test set
    path = os.path.join(data_dir, "test.csv")
    print(f"Reading {path}")
    df = pd.read_csv(path, header=None, sep="\t")
    print(f"Length : {df.shape[0]}")
    df_test = pd.concat([df_test, df])

print("Total train set : " + str(df_train.shape[0]))
print("Total test set : " + str(df_test.shape[0]))

path = os.path.join(out_dir, "train.csv")
df_train.to_csv(path, header=None, index=False, sep="\t")
path = os.path.join(out_dir, "test.csv")
df_test.to_csv(path, header=None, index=False, sep="\t")
