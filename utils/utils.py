import pickle
import yaml
import json
import ast
import subprocess
import time
import os
from unidecode import unidecode
import re
import logging

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_yaml(path):
    with open(path, "r", encoding="utf8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_dict_from_txt(path):
    dictionary = dict()
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for i in range(0,len(lines),2):
            key = lines[i][:-1]
            elem = lines[i+1][:-1]
            try:
                dictionary[key] = ast.literal_eval(elem)
            except:
                print(key)
                print(elem)
                raise
    return dictionary

def keep_alive():
    while True:
        time.sleep(30)
        print("Keep alive...")

def get_new_file_path(dir_path, extension):
    """Get a file name that does not already exists in the specified directory
    and create a path using this name and the given extension.

    Parameters
        dir_path | str
            The directory path where to look for the file names
        extension | str
            The extension that will be used. All files with this extension
            present in the directory must have a name that can be cast into int.
    """
    names = os.listdir(dir_path)
    names = [name for name in names if name[-4:] == extension]
    last_index = -1
    if len(names) > 0:
        last_index = max([int(name[:-4]) for name in names])
    name = str(last_index+1)
    path = os.path.join(dir_path, name + extension)
    return path

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def heavy_normalize_text(text: str) -> str:
    """Perform heavy normalization on text:
        - Remove all accents
        - Lowercase everything
        - Replace all non alphanumeric character by a space
        - Remove duplicate, starting, and trailing space
    """

    # remove accents
    text = unidecode(text)
    # put lowercase
    text = text.lower()
    # replace all non alphanumeric by a space
    text = re.sub(r"[^\w]", " ", text)
    # replace underscore by space
    text = text.replace("_", " ")
    # replace multiple space by a single space
    text = re.sub(r"\s+", " ", text)
    # remove starting or trailing space
    text = text.strip()

    return text

def print_set_sizes(df_train, df_dev, df_test):
    if df_train is not None:
        print(f'Training set size: {df_train.shape[0]}')
    if df_dev is not None:
        print(f'Dev set size: {df_dev.shape[0]}')
    if df_test is not None:
        print(f'Test set size: {df_test.shape[0]}')

def create_logging_config(path):
    """Remove any previous logging configuration and creates a new one

    Parameters
        path - str
            The path to the logging file
    """

    # remove previous logging config if present
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # create new logging config
    logging.basicConfig(filename=path,
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
