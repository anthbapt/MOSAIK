#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from spatialdata.transformations import Affine, set_transformation
import matplotlib.pyplot as plt
import spatialdata as sd
import spatialdata_plot
import seaborn as sns
import scanpy as sc
import pandas as pd

def flipping_local_coordinate(sdata, fov, show=True):
    """
    Applies a vertical flip transformation to the local coordinate system of a specified field of view (FOV).

    This function flips the image and label elements in the Y-axis within the local coordinate system for a
    given FOV by applying an affine transformation. It optionally displays the flipped image, label, and points
    if `show` is True.

    Args:
        sdata: The spatial data object that contains images, labels, and points associated with different FOVs.
        fov (str or int): The identifier of the field of view to flip. If an integer is passed, it will be converted to a string.
        show (bool, optional): If True, the function renders the flipped image, label, and point elements. Defaults to True.

    Returns:
        None

    Raises:
        ValueError: If the specified FOV does not exist in the spatial data.

    Example:
        flipping_local_coordinate(sdata, fov='3', show=True)
    """
    
    if isinstance(fov, int):
        fov = str(fov)

    # Check if the FOV exists in the spatial data
    if fov + "_image" not in sdata.images or fov + "_labels" not in sdata.labels:
        raise ValueError(f"FOV '{fov}' not found in the spatial data.")

    ty = sdata.images[fov + "_image"].shape[-1]
    flipping = Affine(
        [[1, 0, 0],
         [0, -1, ty],
         [0, 0, 1]],
        input_axes=("x", "y"),
        output_axes=("x", "y"))

    set_transformation(sdata.images[fov + "_image"], flipping, to_coordinate_system=fov)
    set_transformation(sdata.labels[fov + "_labels"], flipping, to_coordinate_system=fov)

    if show:
        sdata.pl.render_images(fov + "_image").pl.show(coordinate_systems=[fov])
        sdata.pl.render_labels(fov + "_labels").pl.show(coordinate_systems=[fov])
        sdata.pl.render_points(fov + "_points").pl.show(coordinate_systems=[fov])


def visualise_fov(sdata, coordinate='global'):
    """
    Visualizes the field of view (FOV) data in either the global or local coordinate system.

    This function renders and displays images, labels, points, and shapes for a specific field of view
    in the specified coordinate system. Only 'global' and 'local' are accepted values for the coordinate system.

    Args:
        sdata: The spatial data object containing the FOV data to be visualized. This object is expected to have 
               methods like `pl.render_images`, `pl.render_labels`, `pl.render_points`, and `pl.render_shapes`.
        coordinate (str, optional): The coordinate system for visualizing FOV data. Must be either `'global'` or `'local'`.
                                    Defaults to `'global'`.

    Returns:
        None: This function directly renders and displays the visualizations.

    Raises:
        ValueError: If `coordinate` is not 'global' or 'local'.

    Example:
        visualise_fov(sdata, coordinate='local')
    """

    if coordinate == "global":
        sdata.pl.render_images("global_image").pl.show(coordinate_systems="global")
        sdata.pl.render_labels("global_labels").pl.show(coordinate_systems="global")
        sdata.pl.render_points("global_points").pl.show(coordinate_systems="global")
        sdata.pl.render_shapes("global_shapes").pl.show(coordinate_systems="global")

    elif coordinate == "local":
        sdata.pl.render_images("local_image").pl.show(coordinate_systems="local")
        sdata.pl.render_labels("local_labels").pl.show(coordinate_systems="local")
        sdata.pl.render_points("local_points").pl.show(coordinate_systems="local")
        sdata.pl.render_shapes("local_shapes").pl.show(coordinate_systems="local")

    else:
        raise ValueError("`coordinate` must be either 'global' or 'local'.")