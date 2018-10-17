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
