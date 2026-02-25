#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ========================
# Imports & Setup
# ========================
import os
import warnings
import logging

# Set the working directory where your data and results are located.
# Replace "/path/to/project/" with the absolute or relative path to your project folder.
path = "/path/to/project/"
os.chdir(path)

# Silence unnecessary warnings and set logging to show only warnings or errors.
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io as skio
import skimage.measure as skm
from sbf import visualise_crop
from skimage.transform import AffineTransform, warp

import spatialdata as sd
import spatialdata_plot
import scanpy as sc
import xenium

# ========================
# Paths & Data Loading
# ========================
# zarr_path:
#   Path to the Zarr-formatted spatial dataset from the current path (established before). 
#   If the code is running for the first time you should give a name to the file 
#   and how to access from current path
zarr_path = "ZarrName.zarr"

# slide:
#   Path to the slide-specific directory (often a flattened structure used for annotations,
#   cell labels, or per-slide data). This typically corresponds to a single TMA or slide.
slide = "/Xenium_study_raw_folder"

first_run = user_input = input("Is it the first run (0: False, 1: True): ")

# update=True, if the Xenium is after October 2025 (10X Xenium changes the format of the channels)
if first_run == '1':
    sdata = xenium.xenium(path, update=True)
    sdata.write(zarr_path)

sdata = sd.read_zarr(zarr_path)
adata = sdata.tables["table"]
print(adata.obs.keys())

# ========================
# Basic Spatial Plot
# ========================
xy = adata.obsm['spatial']
plt.scatter(xy[:, 0], xy[:, 1], s=0.0001)

# ========================
# Quality Control Metrics
# ========================
sc.pp.calculate_qc_metrics(adata, percent_top=(10, 20, 50, 150), inplace=True)

# Control probe statistics
cprobes = (adata.obs["control_probe_counts"].sum() / adata.obs["total_counts"].sum() * 100)
cwords = (adata.obs["control_codeword_counts"].sum() / adata.obs["total_counts"].sum() * 100)
print(f"Negative DNA probe count % : {cprobes}")
print(f"Negative decoding count % : {cwords}")

# ========================
# QC Plots
# ========================
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
axs[0].set_title("Total transcripts per cell")
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, ax=axs[1])
axs[1].set_title("Unique transcripts per cell")
sns.histplot(adata.obs["cell_area"], kde=False, ax=axs[2])
axs[2].set_title("Area of segmented cells")
sns.histplot(adata.obs["nucleus_area"] / adata.obs["cell_area"], kde=False, ax=axs[3])
axs[3].set_title("Nucleus ratio")

# ========================
# Filtering & Normalization
# ========================
print("Original dimension: ", adata.shape)
sc.pp.filter_cells(adata, min_counts = 100)
print("Dimension after filtering cells: ", adata.shape)
sc.pp.filter_genes(adata, min_cells = 100)
print("Dimension after filtering genes: ", adata.shape)


adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

# ========================
# Dimensionality Reduction & Clustering
# ========================
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# UMAP & spatial plots
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "leiden"], wspace=0.4, save=True)

adata.obs["x_global_px"] = adata.obsm['spatial'][:,0]
adata.obs["y_global_px"] = adata.obsm['spatial'][:,1]

g = sns.scatterplot(x="x_global_px", y="y_global_px", s=2, marker='.', 
                    data=adata.obs, hue='leiden', palette = "Set2")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
handles, labels = g.get_legend_handles_labels()
for h in handles:
    sizes = h.get_markersize()*8
    h.set_markersize(sizes)
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, ncol=2)
g.set_ylabel("")
g.set_xlabel("")
plt.tight_layout()
plt.savefig('Sample_display_transcripts_cluster.png', format = 'png', dpi = 600)

# ========================
# Visualise specific ROI
# ========================
size = 1500
min_co = [24500, 9000]
max_co = [min_co[0] + size, min_co[1] + size]

visualise_crop(sdata, min_co, max_co)
