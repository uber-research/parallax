from modules.embeddings_utils import low_dimensional_projection
from modules.filter_utils import filter_embeddings_metadata, filter_by_ids, slice_embeddings, filter_embeddings
from modules.formulae_utils import formulae_to_vector


def projection(data_manager, dataset_id=None, data_filters=None, metadata_filters=None, mode=None, rank_slice=None,
               metric=None,
               n_axes=None, formulae=None, items=None, pre_filtering=True, post_filtering=False):
    print("projection", locals())
    # error handling
    if dataset_id is None or not dataset_id in data_manager.dataset_ids:
        raise ValueError("error: dataset_id not specified or out of range")

    if not mode:
        raise ValueError("error: projection_mode not defined")

    if not isinstance(rank_slice, slice):
        if isinstance(rank_slice, (list, tuple)):
            rank_slice = slice(*rank_slice)
        elif rank_slice is None:
            pass
        else:
            raise ValueError("error: rank_slice not valid")

    if mode == "explicit" and (not formulae):
        raise ValueError("error: formulae not defined in the explicit projection_mode")

    if mode == "explicit" and (not metric):
        raise ValueError("error: metric not defined in the explicit projection_mode")

    if mode == "tsne" and (not metric):
        raise ValueError("error: metric not defined in the tsne projection_mode")

    if mode == "tsne" and (not n_axes):
        raise ValueError("error: n_axes not defined in the tsne projection_mode")

    if mode == "pca" and not n_axes:
        raise ValueError("error: n_axes not defined in the pca projection_mode")

    embeddings = data_manager.get_embeddings(dataset_id)
    metadata = data_manager.get_metadata(dataset_id)
    metadata_type = data_manager.get_metadata_type(dataset_id)

    # evaluate formulae
    axes_vectors = None
    if mode == "explicit" and formulae:
        axes_vectors = formulae_to_vector(formulae, embeddings)
        if axes_vectors is None:
            raise ValueError("error: invalid formula or variable not found in formula")

    # evaluate items and add to data manager
    if items and len(items) > 0:
        item_vectors = formulae_to_vector(items, data_manager.get_embeddings(dataset_id))
        if item_vectors is None:
            print("warning: invalid item or variable in item not found")
        else:
            for i, item in enumerate(items):
                if item not in embeddings:
                    embeddings[item] = item_vectors[i]

    # pre filter by metadata
    if pre_filtering:
        embeddings, metadata = filter(embeddings, metadata, metadata_type, rank_slice, metadata_filters, data_filters,
                                      reserved_keys=items, full_embeddings=data_manager.get_embeddings(dataset_id))

    if not embeddings or not len(embeddings) > 0:
        return {}

    # perform projection
    projected_embeddings = low_dimensional_projection(embeddings, mode=mode, metric=metric, axes_vectors=axes_vectors,
                                            n_axes=n_axes)

    # post filter by metadata
    if post_filtering:
        embeddings, metadata = filter(embeddings, metadata, metadata_type, rank_slice, metadata_filters, data_filters,
                                      reserved_keys=items, full_embeddings=data_manager.get_embeddings(dataset_id))
        projected_embeddings = filter_by_ids(projected_embeddings, embeddings.keys())


    result = {}
    for word, coords in projected_embeddings.items():
        val = metadata.get(word, {})
        val["coords"] = coords
        result[word] = val

    return result


def filter(embeddings, metadata, metadata_type, rank_slice=None, metadata_filters=None, data_filters=None,
           reserved_keys=None, full_embeddings=None):
    filtered_embeddings = embeddings
    filtered_metadata = metadata

    # slice
    if rank_slice:
        filtered_ids = slice_embeddings(filtered_embeddings, rank_slice, reserved_keys)
        if len(filtered_ids) == 0:
            return {}, {}
        else:
            filtered_embeddings = filter_by_ids(filtered_embeddings, filtered_ids)
            filtered_metadata = filter_by_ids(filtered_metadata, filtered_ids)

    # metadata filtering
    if metadata_filters and len(metadata_filters) > 0:
        filtered_ids = filter_embeddings_metadata(filtered_metadata,
                                                  metadata_type,
                                                  metadata_filters)
        if len(filtered_ids) == 0:
            return {}, {}
        else:
            filtered_embeddings = filter_by_ids(filtered_embeddings, filtered_ids)
            filtered_metadata = filter_by_ids(filtered_metadata, filtered_ids)

    # data filtering
    if data_filters and len(data_filters) > 0:
        filtered_ids = filter_embeddings(filtered_embeddings,
                                         data_filters,
                                         full_embeddings=full_embeddings if full_embeddings is not None else embeddings)
        if len(filtered_ids) == 0:
            return {}, {}
        else:
            filtered_embeddings = filter_by_ids(filtered_embeddings, filtered_ids)
            filtered_metadata = filter_by_ids(filtered_metadata, filtered_ids)

    return filtered_embeddings, filtered_metadata
