# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict

from modules.embeddings_utils import low_dimensional_projection
from modules.formulae_utils import formulae_to_vector


def filter_by_attribute(metadata_obj, metadata_type, attribute_name, attribute_value):
    """
    three types of metadata attributes:
    1) boolean (value: true or false)
    2) numerical (value: list [min, max])
    3) categorical (value: list [cate1, cate2 ...])

    metadata_obj: {"len", 3, "stopword": False} (for each instance)
    metadata_type: {"len", 3, "stopword": False} (defines the data type of attributes globally)
    """
    try:
        if attribute_name not in metadata_obj:
            raise Exception("attribute_name does not exist")

        attr_type = metadata_type[attribute_name]
        val = metadata_obj[attribute_name]

        if attr_type == 'boolean':
            if type(attribute_value) == bool:
                return attribute_value == val
            elif isinstance(attribute_value, str):
                attribute_value = attribute_value.lower()
                if attribute_value == 'any':
                    return True
                else:
                    return attribute_value == str(val).lower()
            else:
                raise Exception("invalid boolean format")

        if attr_type == 'numerical':
            if isinstance(attribute_value, (list, tuple)) and len(attribute_value) == 2:
                return attribute_value[0] <= val <= attribute_value[1]
            else:
                raise Exception("invalid numerical format")

        if attr_type == 'categorical':
            if isinstance(attribute_value, (set, list, tuple)):
                if len(attribute_value) > 0:
                    return val in attribute_value
                else:
                    return True
            else:
                raise Exception("invalid categorical format")

        if attr_type == 'set':
            if isinstance(attribute_value, (set, list, tuple)):
                if len(attribute_value) > 0:
                    return len(val.intersection(attribute_value)) > 0
                else:
                    return True
            else:
                raise Exception("invalid categorical format")

    except Exception as err:
        print(err)
        return None


def filter_by_attributes(metadata, metadata_type, filter_parameters):
    if not filter_parameters or len(filter_parameters) == 0 or not metadata:
        return True

    for filter_param in filter_parameters:
        if not filter_by_attribute(metadata, metadata_type, filter_param[0], filter_param[1]):
            return False

    return True


def filter_embeddings_metadata(embeddings_metadata, metadata_type, metadata_filters):
    """
    metadata_filters is a list of tuples in this format (attr_name, attr_val)
    """
    filtered_ids = set()
    for id in embeddings_metadata:
        metadata_obj = embeddings_metadata[id]
        if filter_by_attributes(metadata_obj, metadata_type, metadata_filters):
            filtered_ids.add(id)

    return filtered_ids


compare_function_register = {
    'greater': lambda x, y: x > y,
    'greater_equal': lambda x, y: x >= y,
    'close': lambda x, y: x > y - 0.01 and x < y + 0.01,
    'less_equal': lambda x, y: x <= y,
    'less': lambda x, y: x < y
}


def filter_embeddings(embeddings, data_filters, full_embeddings):
    """
    data_filters is a list of dictionaries in this format {measure: measure, formula: formula, compare_function: compare_function, number: number}
    """

    filtered_ids = set(full_embeddings.keys())
    for data_filter in data_filters:
        # evaluate formulae
        axes_vectors = formulae_to_vector([data_filter['formula']], full_embeddings)
        scores = low_dimensional_projection(embeddings, mode='explicit',
                                            metric=data_filter['measure'],
                                            axes_vectors=axes_vectors,
                                            n_axes=1)
        compare_function = compare_function_register[data_filter['compare_function']]
        intersection_ids = set()
        for id, score in scores.items():
            if compare_function(score[0], data_filter['number']) and id in filtered_ids:
                intersection_ids.add(id)
        filtered_ids = intersection_ids

    return filtered_ids


def filter_by_ids(embeddings, ids):
    if not ids:
        return embeddings
    else:
        filtered_embeddings = OrderedDict()
        for id in embeddings:
            if id in ids:
                filtered_embeddings[id] = embeddings[id]
        return filtered_embeddings


def slice_embeddings(embeddings, rank_slice, reserved_keys):
    sliced_keys = set(list(embeddings.keys())[rank_slice])
    for key in reserved_keys:
        sliced_keys.add(key)
    return sliced_keys


if __name__ == '__main__':
    pass
    # dm = DataManager(default_test_datasets, default_embeddings_top_k)
    # filtered = filter_embeddings_metadata(dm.get_metadata(0), dm.get_metadata_type(0), [
    #     ("len", [7, 18]),
    #     ("category", ['A', 'B']),
    #     ("stopword", False),
    # ])
    # print(len(filtered))
