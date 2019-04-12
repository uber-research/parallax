# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from globals import default_test_datasets, default_embeddings_top_k
from modules.data_manager import DataManager


def average(*args):
    return np.mean(np.stack(args), axis=0)


def quantum_not(a, b):
    return a - (np.dot(a, b)) * b


def normalized_quantum_not(a, b):
    # Widdows in "Word Vectors and Quantum Logic Experiments with negation and disjunction" has |b|^2 at the denominator
    # He says that that't b dot b, the norm2 of b, but the norm of b is the root of that.
    # In the proof of Theorem 1 he substitutes |b|^2 with b dot b and the demonstration works,
    # so I implement it a b dot b rather than norm2.
    return a - (np.dot(a, b) / np.dot(b, b)) * b

def cosadd(a, b, c):
    # the analogy function from Mikolov's word2vec paper
    # defined as 3CosSum in Levy & Goldberg's Linguistic Regularities in Sparse and Explicit Word Representations
    return b + c - a

# todo 3CosMul and pair direction

additional_functions = {
    'avg': average,
    'qnot': quantum_not,
    'nqnot': normalized_quantum_not,
    'cosadd': cosadd
}


def formula_to_vector(expression, embeddings):
    return eval(expression, embeddings, additional_functions)


def formulae_to_vector(formulae, embeddings):
    """calculate the vector of the math formulae defined by the user"""
    if not formulae or len(formulae) == 0 or not embeddings or len(embeddings) == 0:
        return None

    results = []
    for formula in formulae:
        try:
            results.append(formula_to_vector(formula, embeddings))
        except Exception as err:
            print(err)
            return None

    return results


if __name__ == '__main__':
    dm = DataManager(default_test_datasets, default_embeddings_top_k)
    dataset_id = dm.dataset_ids[0]

    exprs = [
        "0.5 * apple + 0.5 * technology"
    ]
    print(formulae_to_vector(exprs, dm.get_embeddings(dataset_id)))

    exprs = [
        "avg(apple, technology)"
    ]
    print(formulae_to_vector(exprs, dm.get_embeddings(dataset_id)))

    exprs = [
        "qnot(apple, technology)"
    ]
    print(formulae_to_vector(exprs, dm.get_embeddings(dataset_id)))

    exprs = [
        "nqnot(apple, technology)"
    ]
    print(formulae_to_vector(exprs, dm.get_embeddings(dataset_id)))
