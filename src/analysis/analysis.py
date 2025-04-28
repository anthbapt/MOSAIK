#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from spatialdata_io.experimental import from_legacy_anndata
from spatialdata_io import cosmx
import matplotlib.pyplot as plt
import spatialdata as sd
import scipy.ndimage as scn
from pathlib import Path
import spatialdata_plot
import scanpy as sc
import squidpy as sq
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import glob
import os

path = '/Users/k2481276/Documents/CosMx/SBF_C007/flatFiles/'
Slide1 = '870_SX1930_SX1HV_925'
Slide2 = 'HV_772HV_846HV_843'

flat_file_dir_Slide1 = path + Slide1
flat_file_dir_Slide2 = path + Slide2

meta_file_Slide1 = [item for item in os.listdir(flat_file_dir_Slide1) if 'metadata_file' in item][0]
counts_file_Slide1 = [item for item in os.listdir(flat_file_dir_Slide1) if 'exprMat_file' in item][0]
fov_file_Slide1 = [item for item in os.listdir(flat_file_dir_Slide1) if 'fov_positions_file' in item][0]

columns_sub = ['fov','Area','AspectRatio','CenterX_local_px','CenterY_local_px','Width',
'Height','Mean.PanCK','Max.PanCK','Mean.G','Max.G','Mean.Membrane','Max.Membrane',
'Mean.CD45','Max.CD45','Mean.DAPI','Max.DAPI','cell_ID','assay_type','version',
'Run_Tissue_name','Panel','cellSegmentationSetId','cellSegmentationSetName',
'slide_ID','CenterX_global_px','CenterY_global_px','unassignedTranscripts']


test = pd.read_csv(flat_file_dir_Slide1 + '/' + meta_file_Slide1)
test = test[columns_sub]
test.to_csv(flat_file_dir_Slide1 + '/' + meta_file_Slide1, index = False)

adata_Slide1 = sq.read.nanostring(path = flat_file_dir_Slide1, 
                                  counts_file = counts_file_Slide1,
                                  meta_file = meta_file_Slide1,
                                  fov_file = fov_file_Slide1)

# Calculate quality control metrics
adata_Slide1.var["NegPrb"] = adata_Slide1.var_names.str.startswith("NegPrb")
sc.pp.calculate_qc_metrics(adata_Slide1, qc_vars=["NegPrb"], inplace=True)
pd.set_option("display.max_columns", None)
adata_Slide1.obs["total_counts_NegPrb"].sum() / adata_Slide1.obs["total_counts"].sum() * 100

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].set_title("Total transcripts per cell")
sns.histplot(
    adata_Slide1.obs["total_counts"],
    kde=False,
    ax=axs[0])

axs[1].set_title("Unique transcripts per cell")
sns.histplot(
    adata_Slide1.obs["n_genes_by_counts"],
    kde=False,
    ax=axs[1])

axs[2].set_title("Transcripts per FOV")
sns.histplot(
    adata_Slide1.obs.groupby("fov").sum()["total_counts"],
    kde=False,
    ax=axs[2])


sc.pp.filter_cells(adata_Slide1, min_counts = 100)
sc.pp.filter_genes(adata_Slide1, min_cells = 400)

adata_Slide1.layers["counts"] = adata_Slide1.X.copy()
sc.pp.normalize_total(adata_Slide1, inplace=True)
sc.pp.log1p(adata_Slide1)
sc.pp.pca(adata_Slide1)
sc.pp.neighbors(adata_Slide1)
sc.tl.umap(adata_Slide1)
sc.tl.leiden(adata_Slide1)

sc.pl.umap(
    adata_Slide1,
    color=[
        "total_counts",
        "n_genes_by_counts",
        "leiden",
    ],
    wspace=0.4,
)


sq.pl.spatial_segment(
    adata_Slide1,
    color="Max.PanCK",
    library_key="fov",
    library_id=["1", "83"],
    seg_cell_id="cell_ID",
)


sq.pl.spatial_segment(
    adata_Slide1,
    color="Max.PanCK",
    library_key="fov",
    seg_key = 'segmentation',
    library_id=["1", "83"],
    seg_cell_id="cell_ID",
    seg_contourpx = 10
)


sq.pl.spatial_segment(
    adata_Slide1,
    color="Area",
    library_key="fov",
    library_id=["1", "83"],
    seg_cell_id="cell_ID",
    seg_outline=True,
    cmap="plasma",
    img=False,
    scalebar_dx=1.0,
    scalebar_kwargs={"scale_loc": "bottom", "location": "lower right"},
)


fig, ax = plt.subplots(1, 2, figsize=(15, 7))
sq.pl.spatial_segment(
    adata_Slide1,
    shape="hex",
    color="leiden",
    library_key="fov",
    library_id="1",
    seg_cell_id="cell_ID",
    img=False,
    size=60,
    ax=ax[0],
)

sq.pl.spatial_segment(
    adata_Slide1,
    color="leiden",
    seg_cell_id="cell_ID",
    library_key="fov",
    library_id="83",
    img=False,
    size=60,
    ax=ax[1],
)

# Spatial data
adata_Slide1_sd = cosmx(flat_file_dir_Slide1)

# adata_Slide1_sd.pl.render_images().pl.show(coordinate_systems=["12"])
# adata_Slide1_sd.pl.render_labels().pl.show(coordinate_systems=["12"])
# adata_Slide1_sd.pl.render_points().pl.show(coordinate_systems=["12"])

# adata_Slide1_sd.points["2_points"] = adata_Slide1_sd.points["2_points"].sample(frac=0.01)
# adata_Slide1_sd.pl.render_points().pl.show(coordinate_systems=["12"])


# cat_type = pd.CategoricalDtype(adata_Slide1_sd.points["2_points"]["CellComp"].unique())
# adata_Slide1_sd.points["2_points"]["CellComp"] = adata_Slide1_sd.points["2_points"]["CellComp"].astype(cat_type)

# adata_Slide1_sd.pl.render_points(color="CellComp").pl.show(coordinate_systems=["12"])


def mirror_symetry(x, y, direction='cw'):
    max_coor_y = np.max(y)
    min_coor_y = np.min(y)
    y_new = min_coor_y + (max_coor_y - y)
    
    return x, y_new
    
key_gene = 'COL1A1'
fov_number = '83'
test = adata_Slide1_sd[fov_number + "_points"].compute()
test = test[test['target'] == key_gene]
test_coor_x = np.array(test['x_global_px'])
test_coor_y = np.array(test['y_global_px'])
test_coor_xnew, test_coor_ynew = mirror_symetry(test_coor_x, test_coor_y)
# transform = adata_Slide1_sd["1_labels"].compute().transform['global'].matrix
label = adata_Slide1_sd[fov_number +"_labels"].compute().values
label_binary = scn.binary_closing(label).astype(int)
# label_global = scn.affine_transform(label, transform)
extent = [np.min(test_coor_x), np.max(test_coor_x), np.min(test_coor_y), np.max(test_coor_y)]


plt.scatter(test_coor_x, test_coor_y, s = 0.01)
plt.scatter(test_coor_xnew, test_coor_ynew, s = 0.1)
plt.show()

plt.imshow(label_binary, extent = extent, cmap = 'binary', alpha = 0.5)
plt.scatter(test_coor_xnew, test_coor_ynew, c = 'red', marker = '.', s = 0.5, alpha = 1)
plt.show()

adata_Slide1_sd['1_image']
adata_Slide1_sd.pl.render_images("1_image", cmap="binary").pl.show()
adata_Slide1_sd["1_points"].compute()
adata_Slide1_sd.pl.render_points("1_points").pl.show()
adata_Slide1_sd.pl.render_points("1_points", color="target", groups="IGHG1", palette="red").pl.show()
adata_Slide1_sd['1_shape']


adata_Slide1_sd.pl.render_images().pl.show(coordinate_systems=["1"])
adata_Slide1_sd.pl.render_labels().pl.show(coordinate_systems=["1"])
adata_Slide1_sd.pl.render_points().pl.show(coordinate_systems=["1"])

adata_Slide1_sd.pl.render_images("1_image", cmap = "gray").pl.show()
adata_Slide1_sd.pl.render_points("1_points", cmap = "gray").pl.show()
adata_Slide1_sd.pl.render_labels("1_labels", cmap = "gray").pl.show()


(
    adata_Slide1_sd.pl.render_labels(
        "3_labels",
        outline_alpha=1.0,
        outline_width=0.5
    )
    .pl.render_points(
        "3_points",
        color="target",
        groups="IGHG1",
        palette="cyan"
    )
    .pl.render_points(
        "3_points",
        color="target",
        groups="IGHG2",
        palette="magenta"
    )
    .pl.show(figsize=(8, 8), coordinate_systems = 'global')
)

# stitching cell overlay
fov_position = pd.read_csv(flat_file_dir_Slide1 + '/' + fov_file_Slide1)
img_list = sorted(glob.glob(flat_file_dir_Slide1 + '/CellOverlay/*'))
sample1 = np.arange(1, 61)
sample2 = np.arange(61, 87)
sample3 = np.arange(87, 96)
sample4 = np.arange(96, 107)
samples_list = [sample1, sample2, sample3, sample4]
# Example image positions
samples_with_coords = list()


for l in samples_list:
    images_with_coords = list()
    for k in l:
        x_coord = fov_position[fov_position['fov'] == k]['x_global_px'].values[0]
        y_coord = fov_position[fov_position['fov'] == k]['y_global_px'].values[0]
        temp = (img_list[k-1], (x_coord, y_coord))
        images_with_coords.append(temp)
    samples_with_coords.append(images_with_coords)


# First, calculate the size of the final stitched image
max_width = max(x + Image.open(path).width for path, (x, y) in samples_with_coords[0])
max_height = max(y + Image.open(path).height for path, (x, y) in samples_with_coords[0])

# Create a new blank image (white background, RGB)
stitched_image = Image.new("RGB", (max_width, max_height), color = (255, 255, 255))

# Paste each image at its coordinates
for path, (x, y) in samples_with_coords[0]:
    img = Image.open(path)
    temp = [i[1][1] for i in samples_with_coords[0]]
    max_coor_y = np.max(temp)
    min_coor_y = np.min(temp)
    y_new = min_coor_y + (max_coor_y - y)
    stitched_image.paste(img, (x, y_new))

# Save or show the final result
stitched_image.save("stitched_output_sample1.png")
stitched_image.show()




 
