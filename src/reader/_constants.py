from enum import unique
from spatialdata_io._constants._enum import ModeEnum

@unique
class CosmxKeys(ModeEnum):
    """Keys for *Nanostring Cosmx* formatted dataset."""

    # files and directories
    COUNTS_SUFFIX = "exprMat_file.csv"
    TRANSCRIPTS_SUFFIX = "tx_file.csv"
    METADATA_SUFFIX = "metadata_file.csv"
    POLYGONS_SUFFIX = "polygons.csv"
    FOV_SUFFIX = "fov_positions_file.csv"
    IMAGES_DIR = "CellComposite"
    IMAGES_DIR2 = "Morphology2D"
    LABELS_DIR = "CellLabels"
    # metadata
    FOV = "fov"
    REGION_KEY = "fov_labels"
    INSTANCE_KEY = "cell_ID"
    X_GLOBAL_CELL = "CenterX_global_px"
    Y_GLOBAL_CELL = "CenterY_global_px"
    X_LOCAL_CELL = "CenterX_local_px"
    Y_LOCAL_CELL = "CenterY_local_px"
    X_LOCAL_TRANSCRIPT = "x_local_px"
    Y_LOCAL_TRANSCRIPT = "y_local_px"
    TARGET_OF_TRANSCRIPT = "target"


@unique
class XeniumKeys(ModeEnum):
    """Keys for *10X Genomics Xenium* formatted dataset."""

    # specifications
    XENIUM_SPECS = "experiment.xenium"

    # cell identifiers
    CELL_ID = "cell_id"

    # nucleus and cell boundaries
    NUCLEUS_BOUNDARIES_FILE = "nucleus_boundaries.parquet"
    CELL_BOUNDARIES_FILE = "cell_boundaries.parquet"
    BOUNDARIES_VERTEX_X = "vertex_x"
    BOUNDARIES_VERTEX_Y = "vertex_y"

    # transcripts
    TRANSCRIPTS_FILE = "transcripts.parquet"
    TRANSCRIPTS_X = "x_location"
    TRANSCRIPTS_Y = "y_location"
    TRANSCRIPTS_Z = "z_location"
    QUALITY_VALUE = "qv"
    OVERLAPS_NUCLEUS = "overlaps_nucleus"
    FEATURE_NAME = "feature_name"

    # cell features matrix
    CELL_FEATURE_MATRIX_FILE = "cell_feature_matrix.h5"
    CELL_METADATA_FILE = "cells.parquet"
    CELL_X = "x_centroid"
    CELL_Y = "y_centroid"
    CELL_AREA = "cell_area"
    CELL_NUCLEUS_AREA = "nucleus_area"

    # morphology images
    # before version 2.0.0
    MORPHOLOGY_MIP_FILE = "morphology_mip.ome.tif"
    MORPHOLOGY_FOCUS_FILE = "morphology_focus.ome.tif"
    # from version 2.0.0
    MORPHOLOGY_FOCUS_DIR = "morphology_focus"
    MORPHOLOGY_FOCUS_CHANNEL_IMAGE = "morphology_focus_{:04}.ome.tif"
    # from analysis_summary.html > Image QC of https://www.10xgenomics.com/datasets/preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard
    MORPHOLOGY_FOCUS_CHANNEL_0 = "DAPI"  # nuclear
    MORPHOLOGY_FOCUS_CHANNEL_1 = "ATP1A1/CD45/E-Cadherin"  # boundary
    MORPHOLOGY_FOCUS_CHANNEL_2 = "18S"  # interior - RNA
    MORPHOLOGY_FOCUS_CHANNEL_3 = "AlphaSMA/Vimentin"  # interior - protein

    # post-xenium images
    ALIGNED_IF_IMAGE_SUFFIX = "if_image.ome.tif"
    ALIGNED_HE_IMAGE_SUFFIX = "he_image.ome.tif"

    # image alignment suffix
    ALIGNMENT_FILE_SUFFIX_TO_REMOVE = ".ome.tif"
    ALIGNMENT_FILE_SUFFIX_TO_ADD = "alignment.csv"

    # specs keys
    ANALYSIS_SW_VERSION = "analysis_sw_version"
    # spec which contains xeniumranger version whenever xeniumranger is used to resegment the data; the new, resegmented data
    # needs to be parsed by considering the xeniumranger version
    XENIUM_RANGER = "xenium_ranger"

    # zarr file with labels file and cell summary keys
    CELLS_ZARR = "cells.zarr.zip"
    NUCLEUS_COUNT = "nucleus_count"
    Z_LEVEL = "z_level"

    EXPLORER_SELECTION_X = "X"
    EXPLORER_SELECTION_Y = "Y"
    EXPLORER_SELECTION_KEY = "Selection"


@unique
class VisiumKeys(ModeEnum):
    """Keys for *10X Genomics Visium* formatted dataset."""

    # files and directories
    FILTERED_COUNTS_FILE = "filtered_feature_bc_matrix.h5"
    RAW_COUNTS_FILE = "raw_feature_bc_matrix.h5"

    # images
    IMAGE_HIRES_FILE = "spatial/tissue_hires_image.png"
    IMAGE_LOWRES_FILE = "spatial/tissue_lowres_image.png"

    # scalefactors
    SCALEFACTORS_FILE = "scalefactors_json.json"
    SCALEFACTORS_HIRES = "tissue_hires_scalef"
    SCALEFACTORS_LOWRES = "tissue_lowres_scalef"

    # spots
    SPOTS_FILE_1 = "tissue_positions_list.csv"
    SPOTS_FILE_2 = "tissue_positions.csv"
    SPOTS_X = "pxl_col_in_fullres"
    SPOTS_Y = "pxl_row_in_fullres"

