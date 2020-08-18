# Twitter streamer for Swiss-German text acquisition

This project aims at creating a corpus of Swiss-German sentences using the Twitter API. It relies on *locationIQ*, a free geocoder, to retrieve the geo-localisation of the Twitter users and know where each sentence comes from. The module also relies on *swisstext-bert-lid*, a Swiss-German language identification model, to filter the tweets that are indeed in Swiss-German.

## Installation

### swisstext-bert-lid
*swisstext-bert-lid* needs to be installed manually from

https://github.com/derlin/swisstext-bert-lid

If you encounter the following error :
```zsh
Could not find a version that satisfies the requirement torch>=0.4.1
```
Try to install torch by specifying the link :
```zsh
python -m pip install torch==0.4.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Note that any torch version up to 1.6 will work.

### locationIQ

Also locationiq needs to be installed with

`pip install git+https://github.com/location-iq/locationiq-python-client.git`

### typechecker

To check argument types at runtime, a typechecker is used. This needs to be installed manually from

https://github.com/Karexar/typechecker

### Other modules

All the remaining modules can be installed with

`pip install -r requirements.txt`

## Setup

First you will need credentials both for the twitter api and locationIQ api. They need to be given in a credentials.yaml file in the root directory. A template is available in credentials-template.yaml. Do not forget to rename it.

The streaming need specific Swiss-German words to track Swiss-German tweet. To compute this list of words,  you will need to provide a Swiss-German corpus. It should be a file with one sentence per line. You can copy the corpus from the *swisstext-bert-lid* module, or download one from : https://wortschatz.uni-leipzig.de/en/download/. Note that you may want to combine several GSW corpus to increase the quality of the tracking list. For example, I used SwissCrawl and all GSW corpus available from Leipzig. Without duplicates, it sums up to around 900'000 sentences.   

Optionally, if you choose "specific" or "proportion" (recommended) as "track_word_type" in the config.yaml file, then you will need to provide a bunch of language corpus in the leipzig_32 directory. You can copy the 32 corpus available from Leipzig and used in the *swisstext-bert-lid" module. Each corpus should contains sentences for a given language, and be stored as a text file. The first 3 letters of the name will be taken as language identifier (e.g. name the french corpus 'fre.txt'). The languages correspond to the most common languages with the same alphabet as Swiss-German, and all languages that are very close to Swiss-German, but not Swiss-German (e.g. High german, german dialects...).

Finally, you may want to customize the parameters in the config.yaml.

## How to start  

There are three main processes that are designed to run simultaneously :
 - **stream** : this process listen to tweets in real-time according to the configuration. The corresponding class 'GSW_stream' is defined in 'streamer.py'. A main class is defined in 'scripts/stream.py' and will call the streamer directly.
 ```zsh
 python -m scripts.stream
 ```
 - **search_users** : this process fetches the last 3'200 tweets from the known Swiss-German twitter users identified after running the *stream* and *filter* process for a while. This is very useful to quickly create a dataset, but requires to have identified Swiss-German users. Note that from a first list of Swiss-German users, new users will be found because of nested tweets containing other user ids. The corresponding class is 'GSW_stream' defined in 'streamer.py'. A main class is available in 'scripts/search_users.py'.
 ```zsh
 python -m scripts.search_users
 ```
 -**filter** : this process takes raw tweets fetched by the first two components and process them to extract Swiss-German sentences. In the case the user provided some information about his location, the process will attach a geo-localisation to the sentence. The corresponding class is 'TweetFilter' defined in 'tweet_filter.py'. A main class is available in 'scripts/filter.py'.
 ```zsh
 python -m scripts.filter
 ```

## Notes

- Make sure the time is correct on your machine, otherwise you may encounter a twitter error 401 (unauthorized) when fetching tweets from specific users. Even with correct time, the error may still occur sometimes, and may corresponds to banned users for which we cannot fetch the history.
