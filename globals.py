import os

version = '0.1.0.dev0'

base_dir = os.path.abspath(os.path.dirname(__file__))

default_data_dir = base_dir + "/data"

default_test_datasets = [
    {
        "name": "wikipedia",
        "embeddings_file": default_data_dir + '/glove.6B.50d.txt',
        "metadata_file": default_data_dir + '/glove.6B.50d.metadata.json'
    },
    {
        "name": "twitter",
        "embeddings_file": default_data_dir + '/glove.twitter.27B.50d.txt',
        "metadata_file": default_data_dir + '/glove.twitter.27B.50d.metadata.json'
    },
]

# only load top k word vectors, -1 equals all
default_embeddings_top_k = 10000
