# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import os

version = '0.1.0'

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
