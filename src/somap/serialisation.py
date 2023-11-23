"""Loading and saving functions."""

import equinox as eqx


def save(path_or_file, model):
    """Saves model data to file.

    Args:
        path_or_file: The file location to save values to or a binary file-like object.
        model: The PyTree model whose leaves will be saved
    """
    eqx.tree_serialise_leaves(path_or_file, model)


def load(path_or_file, model_like):
    """Loads model data contained in a file into a `model_like` structure.

    Args:
        path_or_file: The file location to save values to or a binary file-like object.
        model_like: A PyTree model of same structure, and with leaves of the same type,
            as the PyTree being loaded. Those leaves which are loaded will replace the
            corresponding leaves of like.
    """
    return eqx.tree_deserialise_leaves(path_or_file, model_like)
