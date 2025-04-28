"""
Reader function for napari_cosmx plugin
"""
import napari
import os
from napari_cosmx.gemini import Gemini
from ._dock_widget import GeminiQWidget


def napari_get_reader(path):
    """A basic implementation of the napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # Load metadata
    if path.endswith("_metadata.csv"):
        return read_metadata_function
    # Allow parent directory or AnnData file to be opened
    if not (path.endswith(".h5ad") or \
        os.path.exists(os.path.join(path, "images"))):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    if isinstance(path, list):
        # Only want RunSummary directory if list of paths is present
        path = path[0]
    
    viewer = napari.current_viewer()
    gem = Gemini(path, viewer=viewer)
    viewer.window.add_dock_widget(GeminiQWidget(viewer, gem),
        area='right',
        name=gem.name)
    # labels layer added in Gemini instance initialization
    return [(None,)]

def read_metadata_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    if isinstance(path, list):
        path = path[0]
    
    viewer = napari.current_viewer()
    for w in viewer.window._dock_widgets.values():
        if isinstance(w.widget(), GeminiQWidget):
            w.widget().update_metadata(path)
            break
    return [(None,)]