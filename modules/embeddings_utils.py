# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
# coding=utf-8
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def low_dimensional_projection(embeddings, mode="explicit", metric="cosine",
                               axes_vectors=None, n_axes=2, **kwargs):
    """
    projection_mode is one of the three ["pca", "tsne", "explicit"]

    top_k_result determines how many vectors to return
    for pca or tsne modes, we need to calculate all id vectors and get their embedding before returning top k
    for explicit projection_mode, we can just retrieve top k and then calculate the embedding

    for tsne or explicit modes, metrix is the distance measure (not used in pca projection_mode)
    A list of typical distance metrics can be found here:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    for pca or tsne modes, n_axes determines the dimension of the embedding space (not used in explicit projection_mode)
    for explicit projection_mode, axes_val contains a list of vectors representing the coordinates of the explicit axes
    defined by the user in the interface (not used in the pca or tsne modes)
    """
    """Available metrics for scipy cdist: ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, 
    ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, 
    ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule
    These can all be options in the UI, but to begin with let's use: 'cosine', 'correlation' and 'euclidean'
    """
    if not embeddings or len(embeddings) == 0:
        raise Exception("Invalid input parameters")

    if mode == "explicit":
        return _explict_projection(embeddings, metric, axes_vectors, **kwargs)
    elif mode == "tsne":
        return _tsne_projection(embeddings, metric, n_axes, **kwargs)
    elif mode == "pca":
        return _pca_projection(embeddings, n_axes, **kwargs)
    else:
        raise Exception("invalid embedding projection_mode")


def _explict_projection(embeddings, metric, axes_vectors, **kwargs):
    embeddings_matrix = np.stack(embeddings.values())  # num_embeds x hidden
    axes_matrix = np.stack(axes_vectors)  # num_axes x hidden_dimension

    # projected_matrix hs dimensions min(num_ids, top_k) x num_axes
    if metric == 'dot_product':
        projected_matrix = np.matmul(embeddings_matrix, axes_matrix.T)
    elif metric == 'cosine_distance':
        projected_matrix = cdist(embeddings_matrix, axes_matrix,
                                 metric='cosine')
    elif metric == 'cosine_similarity':
        projected_matrix = 1 - cdist(embeddings_matrix, axes_matrix,
                                     metric='cosine')
    else:
        projected_matrix = cdist(embeddings_matrix, axes_matrix, metric=metric)

    projected_emebddings = {embedding_id: projected_matrix[i, :] for
                            i, embedding_id in
                            enumerate(embeddings)}
    return projected_emebddings


def _tsne_projection(embeddings, metric, n_axes, perplexity=30.0,
                     early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                     n_iter_without_progress=300, min_grad_norm=1e-07,
                     init='pca', method='barnes_hut', angle=0.5):
    if metric == 'cosine_distance' or metric == 'cosine_similarity':
        metric = 'cosine'
    # embeddings_matrix = np.stack(list(embeddings.values())[:rank_slice])  # min(num_embeds, top_k) x hidden
    embeddings_matrix = np.stack(embeddings.values())  # num_embeds x hidden
    tsne = TSNE(
        n_components=n_axes,
        metric=metric,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
        n_iter_without_progress=n_iter_without_progress,
        min_grad_norm=min_grad_norm,
        init=init,
        method=method,
        angle=angle,
        verbose=1
    )
    # TSNE doesn't have separate fit and transform, so ccan't be cached...
    start = time.time()
    projected_matrix = tsne.fit_transform(embeddings_matrix)
    print("Took {}s".format(time.time() - start))
    projected_emebddings = {embedding_id: projected_matrix[i, :] for
                            i, embedding_id in
                            enumerate(embeddings)}
    return projected_emebddings


def _pca_projection(embeddings, n_axes, **kwargs):
    # the following line is much faster, but it's mathematically wrong
    # embeddings_matrix = np.stack(list(embeddings.values())[:rank_slice])  # min(num_embeds, top_k) x hidden
    embeddings_matrix = np.stack(embeddings.values())  # num_embeds x hidden
    pca = PCA(n_components=n_axes)
    # this guy may me cached for efficiency, that's why I don;t do fit_transofrm
    pca.fit(embeddings_matrix)
    projected_matrix = pca.transform(embeddings_matrix)
    projected_emebddings = {embedding_id: projected_matrix[i, :] for
                            i, embedding_id in
                            enumerate(embeddings)}
    return projected_emebddings

# if __name__ == '__main__':
#     data_manager = DataManager(default_test_datasets, default_embeddings_top_k)
#     embeddings = data_manager.get_embeddings(data_manager.dataset_ids[0])
#
#     e_proj = low_dimensional_projection(embeddings, mode="explicit", rank_slice=10, metric="cosine",
#                                         axes_vectors=[embeddings['man'], embeddings['woman']])
#     print("Explicit projection:", e_proj)
#
#     pca_proj = low_dimensional_projection(embeddings, mode="pca", rank_slice=10, n_axes=2)
#     print("PCA projection:", pca_proj)
#
#     tsne_proj = low_dimensional_projection(embeddings, mode="tsne", rank_slice=10, metric="euclidean", n_axes=2)
#     print("t-SNE projection:", tsne_proj)
