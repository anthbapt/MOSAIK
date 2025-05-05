#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ========================
# Imports & Setup
# ========================
import os
import warnings
import logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
path = '/Users/k2481276/Documents/CosMx/'
os.chdir(path)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io as skio
import skimage.measure as skm
from sbf import visualise_fov
from skimage.transform import AffineTransform, warp

import spatialdata as sd
from spatialdata_io import xenium
import spatialdata_plot
import squidpy as sq
import scanpy as sc

# ========================
# Paths & Data Loading
# ========================
path = '/Volumes/Xenium/SBF_X001/'
zarr_path = "Xenium.zarr"
os.chdir(path)

first_run = user_input = input("Is it the first run (0: False, 1: True): ")

if first_run == '1':
    sdata = xenium(path)
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
print(adata.shape)
sc.pp.filter_cells(adata, min_counts=100)
print(adata.shape)
sc.pp.filter_genes(adata, min_cells=500)
print(adata.shape)

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
# Neighborhood & Centrality Analysis
# ========================
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
sq.gr.centrality_scores(adata, cluster_key="leiden")
sq.pl.centrality_scores(adata, cluster_key="leiden", figsize=(16, 5))

# ========================
# Subsampling & Neighborhood Enrichment
# ========================
sdata.tables["subsample"] = sc.pp.subsample(adata, fraction=0.1, copy=True)
adata_subsample = sdata.tables["subsample"]

sq.gr.nhood_enrichment(adata, cluster_key="leiden")

fig, ax = plt.subplots(1, 2, figsize=(13, 7))
sq.pl.nhood_enrichment(adata, cluster_key="leiden", figsize=(8, 8), 
                       title="Neighborhood enrichment adata", ax=ax[0])
sq.pl.spatial_scatter(adata_subsample, color="leiden", shape=None, size=2,
                      ax=ax[1], save="leiden2.png", dpi=600)

# ========================
# Spatial Autocorrelation
# ========================
sq.gr.spatial_neighbors(adata_subsample, coord_type="generic", delaunay=True)
sq.gr.spatial_autocorr(adata_subsample, mode="moran", n_perms=100, n_jobs=1)
print(adata_subsample.uns["moranI"].head(10))

# Gene expression spatial plots
sq.pl.spatial_scatter(adata_subsample, library_id="spatial",
                      color=["FOXP3", "NFE2L2", "IL2RA"], shape=None, size=1, 
                      img=False, save="gene_plot.png", dpi=600)

sc.pl.spatial(sdata["table"], color=["FOXP3", "NFE2L2", "IL2RA"], spot_size=100,
              save="gene_plot2.png")

# ========================
# Overlay Gene Expression on Morphology
# ========================
gene_name = ["FOXP3", "NFE2L2", "IL2RA"]
for name in gene_name:
    sdata.pl.render_images("morphology_focus").pl.render_shapes(
        "cell_circles", color=name, table_name="table", use_raw=False
    ).pl.show(
        title=f"{name} expression over Morphology image", 
        coordinate_systems="global", figsize=(10, 5)
    )

# ========================
# Top Spatial Autocorrelated Genes
# ========================
num_view = 12
top_autocorr = adata.uns["moranI"]["I"].sort_values(ascending=False).head(num_view).index.tolist()
sq.pl.spatial_scatter(adata, color=top_autocorr, size=20, cmap="Reds", img=False, figsize=(5, 5))

# ========================
# Align H&E Image with Xenium Data
# ========================        
adata = sdata.tables["table"]    
HE_image = skio.imread("H&E_Run1_Slide1_Sample1.ome.tif", plugin="tifffile")
pixel_size =  0.2125
HE_alignment_matrix = np.genfromtxt("HE_alignment_files/matrix.csv", delimiter=",")
HE_alignment_matrix_inv = np.linalg.inv(HE_alignment_matrix)
adata.obsm['spatial'] = adata.obsm['spatial']/pixel_size
sdata.tables["subsample"] = sc.pp.subsample(adata, fraction=0.05, copy=True)
adata = sdata.tables["subsample"]

adata.obsm['spatial'] = adata.obsm['spatial'].dot(
    HE_alignment_matrix_inv[0:2, 0:2]) + HE_alignment_matrix_inv[0:2, 2]

plt.imshow(HE_image)
plt.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], 
            s=0.1, c='red', alpha=0.6)
plt.savefig('HE_aligned', dpi = 600)

# ========================
# Akoya Image Alignment
# ========================
adata = sdata.tables["table"]
Akoya_image = skio.imread("Akoya_Run1_Slide1_Sample1.ome.tif", plugin="tifffile")
pixel_size =  0.2125
Akoya_alignment_matrix = np.genfromtxt("Akoya_alignment_files/matrix.csv", delimiter=",")
Akoya_alignment_matrix_inv = np.linalg.inv(Akoya_alignment_matrix)
adata.obsm['spatial'] = adata.obsm['spatial']/pixel_size
sdata.tables["subsample"] = sc.pp.subsample(adata, fraction=0.05, copy=True)
adata = sdata.tables["subsample"]

adata.obsm['spatial'] = adata.obsm['spatial'].dot(
    Akoya_alignment_matrix_inv[0:2, 0:2]) + Akoya_alignment_matrix_inv[0:2, 2]

channel = 0
plt.imshow(Akoya_image[channel,:,:])
plt.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], 
            s=0.1, c='red', alpha=0.6)
plt.savefig('Akoya_aligned_channel_' + str(channel), dpi = 600)