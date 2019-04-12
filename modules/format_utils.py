# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

float_precision = 3


def ordered_dict_to_table(embeddings):
    """
    change the format (ordered_dict) of the embeddings to a list of list representation
    change numpy float value to native python float value for serialization
    """
    table = []
    if not embeddings:
        return table

    for word in embeddings:
        table.append([word] + [val.item() for val in np.round(embeddings[word], decimals=float_precision)])

    return table
