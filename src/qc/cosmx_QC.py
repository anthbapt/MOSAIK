#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")

path = '/Users/k2481276/Documents/CosMx/'
os.chdir(path)

from spatialdata.transformations import Affine, set_transformation
import matplotlib.pyplot as plt
import spatialdata as sd
import spatialdata_plot
import seaborn as sns
import scanpy as sc
import pandas as pd
import cosmx


# =============================================================================
# Functions set
# =============================================================================
def flipping_local_coordinate(sdata, fov, show = True):
    if type(fov) == int:
        fov = str(fov)
    ty = sdata.images[fov + "_image"].shape[-1]
    flipping = Affine([[1, 0, 0],
                       [0, -1, ty],
                       [0, 0, 1]], input_axes=("x", "y"), 
                                   output_axes=("x", "y"))
    
    set_transformation(sdata.images[fov + "_image"], flipping, to_coordinate_system = fov)
    set_transformation(sdata.labels[fov + "_labels"], flipping, to_coordinate_system = fov)
    
    if show == True:
        sdata.pl.render_images(fov + "_image").pl.show(coordinate_systems=[fov])
        sdata.pl.render_labels(fov + "_labels").pl.show(coordinate_systems=[fov])
        sdata.pl.render_points(fov + "_points").pl.show(coordinate_systems=[fov])

# fov =  '3'
# flipping_local_coordinate(sdata, fov, show = True)
# =============================================================================
# Paths & Data Loading
# =============================================================================
slide = '/TMAGLACIER2'
zarr_path = "TMAGLACIER2.zarr"

flat_file_dir_slide = path + slide

metafile = [item for item in os.listdir(flat_file_dir_slide) if 'metadata_file' in item][0]
metafile_df = pd.read_csv(flat_file_dir_slide+ '/' + metafile)

fovfile = [item for item in os.listdir(flat_file_dir_slide) if 'fov_positions_file' in item][0]
fovfile_df = pd.read_csv(flat_file_dir_slide + '/' + fovfile)
# =============================================================================
# Uncomment below lines if generating zarr for the first time
# =============================================================================
# columns_sub = ['fov','Area','AspectRatio','CenterX_local_px','CenterY_local_px','Width',
# 'Height','Mean.PanCK','Max.PanCK','Mean.G','Max.G','Mean.Membrane','Max.Membrane',
# 'Mean.CD45','Max.CD45','Mean.DAPI','Max.DAPI','cell_ID','assay_type','version',
# 'Run_Tissue_name','Panel','cellSegmentationSetId','cellSegmentationSetName',
# 'slide_ID','CenterX_global_px','CenterY_global_px','unassignedTranscripts']

# metafile_df = pd.read_csv(flat_file_dir_slide+ '/' + metafile)
# metafile_df = metafile_df[columns_sub]
# metafile_df.to_csv(path + slide + '/' + metafile, index = False)

# fovfile_df = pd.read_csv(flat_file_dir_slide + '/' + fovfile)
# fovfile_df = fovfile_df.rename({'FOV': 'fov'}, axis = 'columns')
# fovfile_df.to_csv(path + slide + '/' + fovfile, index = False)

sdata = cosmx.cosmx(path + slide, type_image = 'composite')
sdata.write(zarr_path)

# =============================================================================
# Data Loading for QC
# =============================================================================
sdata = sd.read_zarr(zarr_path)
adata = sdata.tables["table"]
print(adata.obs.keys())

# FOV positions
sample1 = list(range(1, 5 + 1))
sample2 = list(range(8, 16 + 1))
sample3 = list(range(19, 27 + 1))
sample4 = list(range(33, 37 + 1))
sample5 = list(range(38, 43 + 1))
sample6 = list(range(44, 52 + 1))
sample7 = list(range(53, 61 + 1))
sample8 = list(range(62, 67 + 1))
sample9 = list(range(68, 76 + 1))
sample10 = list(range(77, 85 + 1))
sample11 = list(range(86, 92 + 1))
sample12 = list(range(93, 101 + 1))
sample13 = list(range(102, 110 + 1))
sample14 = list(range(111, 119 + 1))
sample15 = list(range(120, 128 + 1))
sample16 = list(range(130, 133 + 1))
sample17 = list(range(134, 137 + 1))
sample18 = list(range(138, 143 + 1))
sample19 = list(range(144, 152 + 1))
sample20 = list(range(153, 161 + 1))
sample21 = list(range(162, 170 + 1))
sample22 = list(range(171, 179 + 1))
sample23 = list(range(180, 188 + 1))
sample24 = list(range(190, 198 + 1))
sample25 = list(range(199, 207 + 1))
sample26 = list(range(208, 213 + 1))
sample27 = list(range(214, 222 + 1))
sample28 = list(range(223, 231 + 1))
sample29 = list(range(232, 240 + 1))
sample30 = list(range(241, 249 + 1))
sample31 = list(range(250, 258 + 1))
sample32 = list(range(250, 258 + 1))
sample33 = list(range(259, 267 + 1))
sample34 = list(range(268, 276 + 1))
sample35 = list(range(279, 282 + 1))
sample36 = list(range(283, 288 + 1))
sample37 = list(range(289, 297 + 1))
sample38 = list(range(298, 306 + 1))
sample39 = list(range(307, 315 + 1))
sample40 = list(range(316, 324 + 1))
sample41 = list(range(325, 333 + 1))
sample42 = list(range(334, 342 + 1))
sample43 = list(range(343, 351 + 1))
sample44 = list(range(352, 360 + 1))
sample45 = list(range(361, 369 + 1))
sample46 = list(range(370, 378 + 1))
sample47 = list(range(379, 387 + 1))
sample48 = list(range(388, 396 + 1))
sample49 = list(range(397, 405 + 1))
sample50 = [6, 7, 17, 18, 28, 29, 30, 31, 32, 129, 189, 277, 278]

samples_list = [sample1, sample2, sample3, sample4, sample5, sample6, sample7,\
                sample8, sample9, sample10, sample11, sample12, sample13, sample14, \
                sample15, sample16, sample17, sample18, sample19, sample20, sample21, \
                sample22, sample23, sample24, sample25, sample26, sample27, sample28, \
                sample29, sample30, sample31, sample32, sample33, sample34, sample35, \
                sample36, sample37, sample38, sample39, sample40, sample41, sample42, \
                sample43, sample44, sample45, sample46, sample47, sample48, sample49, \
                sample50]
fovfile_df['sampleID'] = pd.Series()
compt = 0
for k in samples_list:
    compt += 1
    for i in k:
        fovfile_df['sampleID'][i-1] = compt


plt.figure(figsize=(11,6))
plt.style.use('default')
plt.grid(False)
g = sns.scatterplot(x="x_global_px", y="y_global_px", s=150, marker='s', 
                    data=fovfile_df, hue='sampleID', palette = "Set2")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
for line in range(0, len(fovfile_df)):
     plt.text(fovfile_df["x_global_px"][line],
              fovfile_df["y_global_px"][line],
              fovfile_df["fov"][line],
              ha='center', fontsize = 5)
handles, labels = g.get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, ncol=3)
plt.tight_layout()
plt.savefig('Sample_display.png', format = 'png', dpi = 600)


g = sns.relplot(x="x_global_px", y="y_global_px", s=75, marker='s', data=fovfile_df,
                hue='sampleID', palette = "Set2", kind="scatter")
for line in range(0, len(fovfile_df)):
     plt.text(fovfile_df["x_global_px"][line],
              fovfile_df["y_global_px"][line],
              fovfile_df["fov"][line],
              ha='center', fontsize = 4)
h,l = g.get_legend_handles_labels()
g.fig.legend(h,l, ncol=2)
plt.savefig('Sample_display.png', format = 'png', dpi = 600)


adata.obs['fov'] = adata.obs['fov'].astype(int)
merged_df = pd.merge(adata.obs, fovfile_df, on='fov', how='left')
adata.obs['sampleID'] = list(merged_df['sampleID'])

# ========================
# Basic Spatial Plot
# ========================
xy = adata.obsm['global']
adata.obs['x_global_px'] = xy[:, 0]
adata.obs['y_global_px'] = xy[:, 1]
plt.scatter(adata.obs['x_global_px'], adata.obs['y_global_px'], s=0.005, marker='.')
plt.savefig('Sample_display_transcripts.png', format = 'png', dpi = 600)

# ========================
# Quality Control Metrics
# ========================
adata.var["NegPrb"] = adata.var_names.str.startswith("Negative")
adata.var["SysControl"] = adata.var_names.str.startswith("SystemControl")

sc.pp.calculate_qc_metrics(adata, qc_vars=["NegPrb"], inplace=True)
sc.pp.calculate_qc_metrics(adata, qc_vars=["SysControl"], inplace=True)

pd.set_option("display.max_columns", None)
negprobes = adata.obs["total_counts_NegPrb"].sum() / adata.obs["total_counts"].sum() * 100
print(f"Negative DNA probes count % : {negprobes}")
syscontrolprobes = adata.obs["total_counts_SysControl"].sum() / adata.obs["total_counts"].sum() * 100
print(f"System Control probes count % : {syscontrolprobes}")

selected_genes = ~adata.var_names.str.contains("SystemControl")
adata = adata[:, selected_genes].copy()

# ========================
# QC Plots Transcripts 
# ========================
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Transcripts Statistics', fontsize=16)
axs[0].set_title("Total transcripts per cell")
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])

axs[1].set_title("Unique transcripts per cell")
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, ax=axs[1])

axs[2].set_title("Area of segmented cells")
sns.histplot(adata.obs["Area"], kde=False, ax=axs[2])
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('statistics.png', format = 'png', dpi = 600)


fig, axs = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Transcripts Statistics FOV', fontsize=16)
axs[0].set_title("Total transcripts per cell")
sns.histplot(adata.obs.groupby("fov")["total_counts"].sum(), kde=False, ax=axs[0])

axs[1].set_title("Unique transcripts per cell")
sns.histplot(adata.obs.groupby("fov")["n_genes_by_counts"].sum(), kde=False, ax=axs[1])

axs[2].set_title("Transcripts per FOV")
sns.histplot(adata.obs.groupby("fov")["total_counts"].sum(), kde=False, ax=axs[2])
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('statistics_FOV.png', format = 'png', dpi = 600)


fig, axs = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Transcripts Statistics sampleID', fontsize=16)
axs[0].set_title("Total transcripts per cell")
sns.barplot(adata.obs.groupby("sampleID")["total_counts"].sum(), ax=axs[0])

axs[1].set_title("Unique transcripts per cell")
sns.barplot(adata.obs.groupby("sampleID")["n_genes_by_counts"].sum(), ax=axs[1])

axs[2].set_title("Transcripts per FOV")
sns.barplot(adata.obs.groupby("sampleID")["total_counts"].sum(), ax=axs[2])
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('statistics_sampleID.png', format = 'png', dpi = 600)

# ========================
# Filtering & Normalization
# ========================
print(adata.shape)
sc.pp.filter_cells(adata, min_counts = 100)
print(adata.shape)
sc.pp.filter_genes(adata, min_cells = 1000)
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
plt.savefig('Sample_display_transcripts2.png', format = 'png', dpi = 600)

    
# visualise specific ROI
fov =  '275'
sdata.pl.render_images(fov + "_image").pl.show(coordinate_systems='global')
sdata.pl.render_labels(fov + "_labels").pl.show(coordinate_systems='global')
sdata.pl.render_points(fov + "_points").pl.show(coordinate_systems='global')
sdata.pl.render_shapes(fov + "_shapes").pl.show(coordinate_systems='global')
