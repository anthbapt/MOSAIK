# This code is an extension of the readers from https://github.com/scverse/spatialdata-io/tree/main
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any
import pyarrow.parquet as pq
from collections import Counter

import dask.array as da
import numpy as np
import pandas as pd
import pyarrow as pa
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from dask_image.imread import imread
from scipy.sparse import csr_matrix
from skimage.transform import estimate_transform
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Affine, Identity
from shapely.geometry import Point, Polygon
from _constants import CosmxKeys

def cosmx(
    path: str | Path,
    dataset_id: str | None = None,
    transcripts: bool = True,
    polygons: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    flip_image: bool = False,
    type_image: str = 'composite') -> SpatialData:
    """Read Nanostring CosMx Spatial Molecular Imager dataset and return a SpatialData object.

    This function reads multiple data components (counts, metadata, images, labels, transcripts, polygons)
    from a CosMx dataset and assembles them into a `SpatialData` object with correct transformations
    and spatial metadata.

    Args:
        path (str | Path): Path to the root directory containing CosMx files.
        dataset_id (str | None, optional): Identifier of the dataset. If None, inferred from filenames.
        transcripts (bool, optional): Whether to read transcript data (default is True).
        polygons (bool, optional): Whether to read and parse polygon data (default is True).
        imread_kwargs (Mapping[str, Any], optional): Extra keyword arguments for image reading via `imread`.
        image_models_kwargs (Mapping[str, Any], optional): Extra keyword arguments for image model parsing.
        flip_image (bool, optional): Whether to vertically flip images and labels (default is False).
        type_image (str, optional): Image type to load, either "composite" or "morphology" (default is "composite").

    Returns:
        SpatialData: A `SpatialData` object containing images, labels, points (transcripts), shapes (polygons), and table data.

    Raises:
        ValueError: If the `dataset_id` cannot be inferred.
        FileNotFoundError: If any of the required files or directories are missing.

    See Also:
        https://nanostring.com/products/cosmx-spatial-molecular-imager/
    """

    path = Path(path)

    # tries to infer dataset_id from the name of the counts file
    if dataset_id is None:
        counts_files = [f for f in os.listdir(path) if str(f).endswith(CosmxKeys.COUNTS_SUFFIX)]
        if len(counts_files) == 1:
            found = re.match(rf"(.*)_{CosmxKeys.COUNTS_SUFFIX}", counts_files[0])
            if found:
                dataset_id = found.group(1)
                
    if dataset_id is None:
        raise ValueError("Could not infer `dataset_id` from the name of the counts file. Please specify it manually.")

    # check for file existence
    counts_file = path / f"{dataset_id}_{CosmxKeys.COUNTS_SUFFIX}"
    
    if not counts_file.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_file}.")
        
    if transcripts:
        transcripts_file = path / f"{dataset_id}_{CosmxKeys.TRANSCRIPTS_SUFFIX}"
        
        if not transcripts_file.exists():
            raise FileNotFoundError(f"Transcripts file not found: {transcripts_file}.")
    else:
        transcripts_file = None
        
    if polygons:
        polygons_file = path / f"{dataset_id}_{CosmxKeys.POLYGONS_SUFFIX}"
        if not polygons_file.exists():
            raise FileNotFoundError(f"Polygons file not found: {polygons_file}.")
    else:
        polygons_file = None
    meta_file = path / f"{dataset_id}_{CosmxKeys.METADATA_SUFFIX}"
    
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}.")
    fov_file = path / f"{dataset_id}_{CosmxKeys.FOV_SUFFIX}"
    
    if not fov_file.exists():
        raise FileNotFoundError(f"Found field of view file: {fov_file}.")
    
    if type_image == "composite":
        images_dir = path / CosmxKeys.IMAGES_DIR
    elif type_image == "morphology":
        images_dir = path / CosmxKeys.IMAGES_DIR2
        
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}.")
    labels_dir = path / CosmxKeys.LABELS_DIR
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}.")

    counts = pd.read_csv(path / counts_file, header=0, index_col=CosmxKeys.INSTANCE_KEY)
    counts.index = counts.index.astype(str).str.cat(counts.pop(CosmxKeys.FOV).astype(str).values, sep="_")

    obs = pd.read_csv(path / meta_file, header=0, index_col=CosmxKeys.INSTANCE_KEY)
    obs[CosmxKeys.FOV] = pd.Categorical(obs[CosmxKeys.FOV].astype(str))
    obs[CosmxKeys.REGION_KEY] = pd.Categorical(obs[CosmxKeys.FOV].astype(str).apply(lambda s: s + "_labels"))
    obs[CosmxKeys.INSTANCE_KEY] = obs.index.astype(np.int64)
    obs.rename_axis(None, inplace=True)
    obs.index = obs.index.astype(str).str.cat(obs[CosmxKeys.FOV].values, sep="_")

    common_index = obs.index.intersection(counts.index)

    adata = AnnData(
        csr_matrix(counts.loc[common_index, :].values),
        dtype=counts.values.dtype,
        obs=obs.loc[common_index, :])
    adata.var_names = counts.columns

    table = TableModel.parse(
        adata,
        region=list(set(adata.obs[CosmxKeys.REGION_KEY].astype(str).tolist())),
        region_key=CosmxKeys.REGION_KEY.value,
        instance_key=CosmxKeys.INSTANCE_KEY.value)

    fovs_counts = list(map(str, adata.obs.fov.astype(int).unique()))
    # we remove all the FOV that contain only one transcripts, because we cannot
    # determine the transfromation based on one pair of coordinate
    my_dict = Counter(obs['fov'])
    keys_with_value = [k for k, v in my_dict.items() if v == 1]
    fovs_counts = sorted(list(set(fovs_counts) - set(keys_with_value)))
    affine_transforms_to_global = {}
    
    for fov in fovs_counts:

        idx = table.obs.fov.astype(str) == fov
        loc = table[idx, :].obs[[CosmxKeys.X_LOCAL_CELL, CosmxKeys.Y_LOCAL_CELL]].values
        glob = table[idx, :].obs[[CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL]].values
        out = estimate_transform(ttype="affine", src=loc, dst=glob)
        affine_transforms_to_global[fov] = Affine(
            # out.params, input_coordinate_system=input_cs, output_coordinate_system=output_cs
            out.params,
            input_axes=("x", "y"),
            output_axes=("x", "y"))

    table.obsm["global"] = table.obs[[CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL]].to_numpy()
    table.obsm["spatial"] = table.obs[[CosmxKeys.X_LOCAL_CELL, CosmxKeys.Y_LOCAL_CELL]].to_numpy()
    table.obs.drop(
        columns=[CosmxKeys.X_LOCAL_CELL, CosmxKeys.Y_LOCAL_CELL, CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL],
        inplace=True)

    # prepare to read images and labels
    file_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff", "TIF")
    pat = re.compile(r".*_F(\d+)")

    # check if fovs are correct for images and labels
    fovs_images = []
    
    if type_image == "composite":
        for fname in os.listdir(path / CosmxKeys.IMAGES_DIR):
            
            if fname.endswith(file_extensions):
                fovs_images.append(str(int(pat.findall(fname)[0])))

    elif type_image == "morphology":
        for fname in os.listdir(path / CosmxKeys.IMAGES_DIR2):
            
            if fname.endswith(file_extensions):
                fovs_images.append(str(int(pat.findall(fname)[0])))

    fovs_labels = []
    for fname in os.listdir(path / CosmxKeys.LABELS_DIR):
        
        if fname.endswith(file_extensions):
            
            fovs_labels.append(str(int(pat.findall(fname)[0])))

    fovs_images_and_labels = set(fovs_images).intersection(set(fovs_labels))
    fovs_diff = fovs_images_and_labels.difference(set(fovs_counts))
    
    if len(fovs_diff):
        logger.warning(
            f"Found images and labels for {len(fovs_images)} FOVs, but only {len(fovs_counts)} FOVs in the counts file.\n"
            + f"The following FOVs are missing: {fovs_diff} \n"
            + "... will use only fovs in Table.")

    # read images
    print("Read Images")
    images = {}
    
    if type_image == "composite":
        
        for fname in os.listdir(path / CosmxKeys.IMAGES_DIR):
            
            if fname.endswith(file_extensions):
                fov = str(int(pat.findall(fname)[0]))
                
                if fov in fovs_counts:
                    aff = affine_transforms_to_global[fov]
                    im = imread(path / CosmxKeys.IMAGES_DIR / fname, **imread_kwargs).squeeze()
                    
                    if flip_image == True:
                        print("Images flipped")
                        flipped_im = da.flip(im, axis=0)
                        parsed_im = Image2DModel.parse(
                            flipped_im,
                            transformations={
                                fov: Identity(),
                                "global": aff,
                                "global_only_image": aff},
                            dims=("y", "x", "c"),
                            rgb=None,
                            **image_models_kwargs)
                        images[f"{fov}_image"] = parsed_im
                    else:
                        parsed_im = Image2DModel.parse(
                            im,
                            transformations={
                                fov: Identity(),
                                "global": aff,
                                "global_only_image": aff},
                            dims=("y", "x", "c"),
                            rgb=None,
                            **image_models_kwargs)
                        images[f"{fov}_image"] = parsed_im
                        
                else:
                    logger.warning(f"FOV {fov} not found in counts file. Skipping image {fname}.")
    elif type_image == "morphology":
        
        for fname in os.listdir(path / CosmxKeys.IMAGES_DIR2):
            
            if fname.endswith(file_extensions):
                fov = str(int(pat.findall(fname)[0]))
                
                if fov in fovs_counts:
                    aff = affine_transforms_to_global[fov]
                    im = imread(path / CosmxKeys.IMAGES_DIR2 / fname, **imread_kwargs).squeeze()
                    
                    if flip_image == True:
                        print("Images flipped")
                        flipped_im = da.flip(im, axis=0)
                        parsed_im = Image2DModel.parse(
                            flipped_im,
                            transformations={
                                fov: Identity(),
                                "global": aff,
                                "global_only_image": aff},
                            dims=("c", "y", "x"),
                            rgb=None,
                            **image_models_kwargs)
                        images[f"{fov}_image"] = parsed_im
                    else:
                        parsed_im = Image2DModel.parse(
                            im,
                            transformations={
                                fov: Identity(),
                                "global": aff,
                                "global_only_image": aff},
                            dims=("c", "y", "x"),
                            rgb=None,
                            **image_models_kwargs)
                        images[f"{fov}_image"] = parsed_im
                        
                else:
                    logger.warning(f"FOV {fov} not found in counts file. Skipping image {fname}.")

    # read labels
    print("Read Labels")
    labels = {}
    
    for fname in os.listdir(path / CosmxKeys.LABELS_DIR):
        
        if fname.endswith(file_extensions):
            fov = str(int(pat.findall(fname)[0]))
            
            if fov in fovs_counts:
                aff = affine_transforms_to_global[fov]
                la = imread(path / CosmxKeys.LABELS_DIR / fname, **imread_kwargs).squeeze()
                
                if flip_image == True:
                    print("Labels flipped")
                    flipped_la = da.flip(la, axis=0)
                    parsed_la = Labels2DModel.parse(
                        flipped_la,
                        transformations={
                            fov: Identity(),
                            "global": aff,
                            "global_only_labels": aff},
                        dims=("y", "x"),
                        **image_models_kwargs)
                    labels[f"{fov}_labels"] = parsed_la
                else:
                    parsed_la = Labels2DModel.parse(
                        la,
                        transformations={
                            fov: Identity(),
                            "global": aff,
                            "global_only_labels": aff},
                        dims=("y", "x"),
                        **image_models_kwargs)
                    labels[f"{fov}_labels"] = parsed_la
            else:
                logger.warning(f"FOV {fov} not found in counts file. Skipping labels {fname}.")

    # read polygons
    print("Read polygons")
    shapes: dict[str, GeoDataFrame] = {}
    
    if polygons:
        
        with tempfile.TemporaryDirectory() as tmpdir:
            assert polygons_file is not None
            polygons_data = pd.read_csv(path / polygons_file, header=0)
            counter = Counter(polygons_data['cell'])
            cell_bad_polygons = [i for i, count in counter.items() if count < 4]
            polygons_data = polygons_data[~polygons_data["cell"].isin(cell_bad_polygons)]
            polygons_data["geometry"] = polygons_data.apply(lambda row: Point(row["x_local_px"], row["y_local_px"]), axis=1)
            polygons = polygons_data.groupby("cell")["geometry"].apply(lambda points: Polygon([(p.x, p.y) for p in points]))
            
            geo_df = GeoDataFrame(polygons, geometry=polygons)
            geo_df = geo_df.reset_index()
            geo_df["centroid"] = geo_df.centroid
            geo_df["radius"] = (geo_df['geometry'].area)/(2*np.pi)
            cell_to_fov = {polygons_data['cell'].iloc[i]:polygons_data['fov'].iloc[i] for i in range(len(polygons_data))}
            geo_df['fov'] = geo_df['cell'].map(cell_to_fov)
            
            for fov in fovs_counts:
                aff = affine_transforms_to_global[fov]
                sub_geo_df = geo_df[geo_df['fov'] == int(fov)]

                if len(sub_geo_df) > 0:
                    shapes[f"{fov}_shapes"] = ShapesModel.parse(sub_geo_df,
                                                                transformations={
                                                                    fov: Identity(),
                                                                    "global": aff,
                                                                    "global_only_labels": aff})

    # read transcripts
    print("Read transcripts")
    points: dict[str, DaskDataFrame] = {}
    
    if transcripts:
        
        with tempfile.TemporaryDirectory() as tmpdir:
            print("converting .csv to .parquet to improve the speed of the slicing operations... ", end="")
            assert transcripts_file is not None
            transcripts_data = pd.read_csv(path / transcripts_file, header=0)
            transcripts_data.to_parquet(Path(tmpdir) / "transcripts.parquet")
            print("done")
            ptable = pq.read_table(Path(tmpdir) / "transcripts.parquet")
            
            for fov in fovs_counts:
                aff = affine_transforms_to_global[fov]
                sub_table = ptable.filter(pa.compute.equal(ptable.column(CosmxKeys.FOV), int(fov))).to_pandas()
                sub_table[CosmxKeys.INSTANCE_KEY] = sub_table[CosmxKeys.INSTANCE_KEY].astype("category")
                # we rename z because we want to treat the data as 2d
                sub_table.rename(columns={"z": "z_raw"}, inplace=True)
                
                if len(sub_table) > 0:
                    points[f"{fov}_points"] = PointsModel.parse(
                        sub_table,
                        coordinates={"x": CosmxKeys.X_LOCAL_TRANSCRIPT, "y": CosmxKeys.Y_LOCAL_TRANSCRIPT},
                        feature_key=CosmxKeys.TARGET_OF_TRANSCRIPT,
                        instance_key=CosmxKeys.INSTANCE_KEY,
                        transformations={
                            fov: Identity(),
                            "global": aff,
                            "global_only_labels": aff})

    return SpatialData(images=images, labels=labels, points=points, shapes=shapes, table=table)
