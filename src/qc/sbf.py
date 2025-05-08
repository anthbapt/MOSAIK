#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from spatialdata.transformations import Affine, set_transformation
from spatialdata import bounding_box_query
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


def visualise_fov(sdata, fov, coordinate='global'):
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
    
    if isinstance(fov, int):
        fov = str(fov)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))    

    if coordinate == "global":
        ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sdata.pl.render_images(fov + "_image").pl.show(coordinate_systems="global", title="Images", dpi = 600, ax=ax[0])
        ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sdata.pl.render_labels(fov + "_labels").pl.show(coordinate_systems="global", title="Labels", dpi = 600, ax=ax[1])
        ax[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sdata.pl.render_points(fov + "_points", size = 0.0001).pl.show(coordinate_systems="global", title="Points", dpi = 600, ax=ax[2])
        ax[3].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sdata.pl.render_shapes(fov + "_shapes", scale=0.9).pl.show(coordinate_systems="global", title="Shapes", dpi = 600, ax=ax[3])
        plt.savefig('CosMx.png', format = 'png', dpi = 600)
        
    elif coordinate == "local":
        ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sdata.pl.render_images(fov + "_image").pl.show(coordinate_systems=fov, title="Images", dpi = 600, ax=ax[0])
        ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sdata.pl.render_labels(fov + "_labels").pl.show(coordinate_systems=fov, title="Labels", dpi = 600, ax=ax[1])
        ax[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sdata.pl.render_points(fov + "_points", size = 0.0001).pl.show(coordinate_systems=fov, title="Points", dpi = 600, ax=ax[2])
        ax[3].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sdata.pl.render_shapes(fov + "_shapes", scale=0.9).pl.show(coordinate_systems=fov, title="Shapes", dpi = 600, ax=ax[3])
        plt.savefig('CosMx.png', format = 'png', dpi = 600)

    else:
        raise ValueError("`coordinate` must be either 'global' or 'local'.")
        
        
def crop(x, min_co, max_co):
    """
   Crops a spatial object to a specified bounding box.

   Applies a bounding box query on the input `x`, limiting the data to the given 
   minimum and maximum coordinates along the x and y axes in the global coordinate system.

   Args:
       x: The spatial data object to be cropped. This is typically a SpatialData 
          object or any compatible format supported by `bounding_box_query`.
       min_co: The minimum coordinate (e.g., tuple or list) defining one corner of 
               the bounding box in the format (x_min, y_min).
       max_co: The maximum coordinate (e.g., tuple or list) defining the opposite 
               corner of the bounding box in the format (x_max, y_max).

   Returns:
       A spatial data object cropped to the specified bounding box.

   """
   
    return bounding_box_query(x, min_coordinate=min_co,
                              max_coordinate=max_co, axes=("x", "y"),
                              target_coordinate_system="global")    
    
def visualise_crop(sdata, min_co, max_co):
    """
    Visualizes a cropped region of spatial data across multiple modalities.

    This function generates a 1x4 subplot showing the output of four different
    spatial data layers: morphology images, cell labels, transcript points, and
    cell boundaries. The visualized region is defined by a bounding box specified
    by `min_co` and `max_co`.

    Args:
        sdata: A `SpatialData` object containing spatial transcriptomics data and 
               associated modalities (e.g., images, labels, points, shapes).
        min_co: The minimum coordinate (e.g., tuple or list) for cropping, 
                typically in the format (x_min, y_min).
        max_co: The maximum coordinate (e.g., tuple or list) for cropping, 
                typically in the format (x_max, y_max).

    Returns:
        None. The function displays the plots and saves them as 'Xenium.png'.

    Side Effects:
        - Displays a matplotlib figure with 4 subplots.
        - Saves the figure as a PNG file named 'Xenium.png' in the current directory.

    """

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))    

    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    crop(sdata, min_co, max_co).pl.render_images("morphology_focus").pl.show(coordinate_systems="global", title="Images", dpi = 600, ax=ax[0])
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    crop(sdata, min_co, max_co).pl.render_labels("cell_labels").pl.show(coordinate_systems="global", title="Labels", dpi = 600, ax=ax[1])
    ax[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    crop(sdata, min_co, max_co).pl.render_points("transcripts", size = 0.00001).pl.show(coordinate_systems="global", title="Points", dpi = 600, ax=ax[2])
    ax[3].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    crop(sdata, min_co, max_co).pl.render_shapes("cell_boundaries", scale=0.9).pl.show(coordinate_systems="global", title="Shapes", dpi = 600, ax=ax[3])
    plt.savefig('Xenium.png', format = 'png', dpi = 600)