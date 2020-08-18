This folder contains the cleaning scripts to use on the generated twitter dataset (i.e. 'gsw_sentences.csv' and 'gsw_tweets.csv' in the final_dataset folder).

##### 1_map_coords_to_state.py

This  takes the coordinates for each sentence and map it to the swiss canton using the reverse geocoding functionality of locationIQ. It saves the dict on disk. If the state is not in switzerland, or not available, then the label is None.

##### 2_check_location.py

This is to perform a manual check of the locations. The user is prompted with the user.location field (i.e. given by the twitter user on his account), and the state to which it is mapped. The user needs to tell wether or not the location has enough information to infer a location at least at the state (canton) level. Generally, if the answer is yes, then the mapping is correct. In some case (~1-2%), there are enough information, but the mapping given by locationIQ is incorrect. In this case the user can correct the mapping.

##### 3_clean.py

Use the state information to map each location to the given dialect. There is also a filtering part to take only sentences where the lid prediction is above a given threshold. The output is a tab-separated .csv file with no header, containing two columns : the sentences and the corresponding dialect.
