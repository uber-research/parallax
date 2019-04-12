# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from collections import OrderedDict

import numpy as np

from globals import default_test_datasets, default_embeddings_top_k


class DataManager(object):

    def __init__(self, datasets, top_k=-1):
        """
        wordVectors is a sorted dict of words with its coordinates
        wordMetadata is a dict of words with associated metadata
        """
        # self.embeddings = OrderedDict()
        # self.embeddings_metadata = {}
        # self.embeddings_metadata_type = {}
        if not datasets or len(datasets) == 0:
            raise Exception("error: invalid datasets")

        self.embeddings_data = {}
        for dataset in datasets:
            print('Loading dataset', dataset['name'])
            self.embeddings_data[dataset['name']] = self._load_dataset(dataset["embeddings_file"],
                                                                       dataset.get("metadata_file", None), top_k)
            print()

        self.dataset_ids = [dataset['name'] for dataset in datasets]

    def _load_dataset(self, embeddings_file, metadata_file, top_k=-1):
        """load txt file, each line contains a word vector with the format: word dim1 dim2 ..."""

        embeddings = OrderedDict()
        embeddings_metadata = {}
        embeddings_metadata_type = {}
        embeddings_metadata_domain = {}

        with open(embeddings_file, "r", encoding="utf-8") as data:
            for idx, datum in enumerate(data):

                if top_k != -1 and idx >= top_k:
                    break

                l = datum.split(" ")
                embedding = str(l[0])
                embeddings[embedding] = np.array([float(val) for val in l[1:]])

            print("{} embeddings loaded in memory from file: {}".format(len(embeddings), embeddings_file))

        if metadata_file is not None:
            try:
                with open(metadata_file, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    embeddings_metadata_type = data["types"]
                    metadata = data["values"]

                    # init domain
                    for attribute, attribute_type in embeddings_metadata_type.items():
                        if attribute_type == "numerical":
                            embeddings_metadata_domain[attribute] = [1e12, -1e12]
                        elif attribute_type == "categorical":
                            embeddings_metadata_domain[attribute] = set()
                        elif attribute_type == "boolean":
                            embeddings_metadata_domain[attribute] = set()
                        elif attribute_type == "set":
                            embeddings_metadata_domain[attribute] = set()
                        else:
                            print("attribute type: {} not supported yet".format(attribute_type))

                    for embedding in embeddings:
                        if embedding in metadata:
                            embeddings_metadata[embedding] = metadata[embedding]
                            # update domain
                            for attribute, value in metadata[embedding].items():
                                attribute_type = embeddings_metadata_type[attribute]
                                if attribute_type == "numerical":
                                    embeddings_metadata_domain[attribute] = [
                                        min(embeddings_metadata_domain[attribute][0], value),
                                        max(embeddings_metadata_domain[attribute][1], value)
                                    ]
                                elif attribute_type == "categorical":
                                    embeddings_metadata_domain[attribute].add(value)
                                elif attribute_type == "boolean":
                                    embeddings_metadata_domain[attribute].add(value)
                                elif attribute_type == "set":
                                    value = set(value)
                                    embeddings_metadata[embedding][attribute] = value
                                    embeddings_metadata_domain[attribute].update(value)
                                else:
                                    print("Metadata attribute type not supported")
                        else:
                            print("Cannot find the metadata of embedding {} in file: {}".format(embedding, metadata_file))

                    print("Metadata of {} embeddings loaded in memory from file: {}".format(len(embeddings_metadata),
                                                                                            metadata_file))
                    print('Metadata type:', embeddings_metadata_type)
                    print('Metadata domain:', embeddings_metadata_domain)
            except FileNotFoundError:
                print('File {} not found, skipping loagind metadata')

        return {
            "embeddings": embeddings,
            "embeddings_metadata": embeddings_metadata,
            "embeddings_metadata_type": embeddings_metadata_type,
            "embeddings_metadata_domain": embeddings_metadata_domain
        }

    def get_embeddings(self, dataset_id):
        return self.embeddings_data[dataset_id]["embeddings"]

    def get_metadata(self, dataset_id):
        return self.embeddings_data[dataset_id]["embeddings_metadata"]

    def get_metadata_type(self, dataset_id):
        return self.embeddings_data[dataset_id]["embeddings_metadata_type"]

    def get_metadata_domain(self, dataset_id):
        return self.embeddings_data[dataset_id]["embeddings_metadata_domain"]

    def get_size(self, dataset_id):
        return len(self.embeddings_data[dataset_id]["embeddings"])

    def get_num_datasets(self):
        return len(self.embeddings_data)


if __name__ == '__main__':
    dm = DataManager(default_test_datasets, default_embeddings_top_k)
