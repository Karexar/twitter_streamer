import pandas as pd
import os
from sklearn.model_selection import train_test_split

###  SETTINGS  ################################################################
input_dir = "data/whatsup/labelled"
file_name = "full.csv"
delimiter = "\t"
header = None
columns=["sentence", "label"]
output_dir = "data/whatsup/labelled"
test_size = 0.1
###############################################################################

# Load the source dataset
data_path = os.path.join(input_dir, file_name)
df = pd.read_csv(data_path, delimiter=delimiter, header=header, names=columns)
print("Dataset shape : " + str(df.shape))
dialects = set(df["label"])

for dialect in dialects:
    group_indices = df[df["label"] == dialect].index.values.tolist()
    train, test = train_test_split(group_indices,
                                   test_size = test_size,
                                   random_state = 42)

    df.at[train, 'set_type'] = 'train'
    df.at[test, 'set_type'] = 'test'
    print(f'Split for {dialect}: {len(train)} train / {len(test)} test / ' +
          f'{len(group_indices)} total')

train_df = df[df['set_type'] == 'train'].drop(labels=['set_type'], axis=1)
test_df = df[df['set_type'] == 'test'].drop(labels=['set_type'], axis=1)
print("train_df shape : " + str(train_df.shape))
print("test_df shape : " + str(test_df.shape))

train_df.to_csv(os.path.join(output_dir, "train.csv"),
                sep="\t",
                index=False,
                header=None)
test_df.to_csv(os.path.join(output_dir, "test.csv"),
               sep="\t",
               index=False,
               header=None)
