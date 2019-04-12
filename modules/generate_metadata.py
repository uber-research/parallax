# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

from globals import default_test_datasets
from modules.data_manager import DataManager

stopwords = set(stopwords.words('english'))

metadata_type = {
    "length": "numerical",
    "pos tag": "set",
    "stopword": "boolean"
}

to_pos = {
    'a': 'Adjective',
    's': 'Adjective Sat',
    'r': 'Adverb',
    'n': 'Noun',
    'v': 'Verb',
}


def generate_postag(word):
    poss = set()
    try:
        synsets = wn.synsets(word)
        for synset in synsets:
            poss.add(to_pos[synset.pos()])
    except:
        print(word + " dosn't have synsets")
    return list(poss)


def is_stopword(word):
    return word.lower() in stopwords


def generate_metadata_for_single_word(word):
    return {
        "length": len(word),
        "pos tag": generate_postag(word),
        "stopword": is_stopword(word),
    }


def generate_metadata(embeddings):
    """
    generate some sample attributes for each word
    these attributes are from additional resources and may not be relevant to the word vectors
    """
    metadata = {}
    for word in embeddings:
        metadata[word] = generate_metadata_for_single_word(word)

    return {"types": metadata_type, "values": metadata}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasets', type=json.loads, default=json.dumps(default_test_datasets),
                        help='loads custom embeddings. It accepts a JSON string containing a list of dictionaries. '
                             'Each dictionary should contain a name field, an embedding_file filed '
                             'and a metadata_file field. '
                             'For example: \'[{"name": "wikipedia", "embedding_file": "...", "metadata_file": "..."}, '
                             '{"name": "twitter", "embedding_file": "...", "metadata_file": "..."}]\'')
    args = parser.parse_args()
    datasets = args.datasets

    data_manager = DataManager(datasets, -1)

    for i, dataset_id in enumerate(data_manager.dataset_ids):
        metadata = generate_metadata(data_manager.get_embeddings(dataset_id))

        with open(datasets[i]["metadata_file"], "w") as output:
            json.dump(metadata, output)
