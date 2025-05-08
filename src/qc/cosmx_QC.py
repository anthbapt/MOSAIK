# =============================================================================
# 
# QuarterBrain
#
# =============================================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings
import logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
path = r"/Volumes/Extreme SSD/QuarterBrain"
os.chdir(path)

from spatialdata.transformations import Affine, set_transformation
import matplotlib.pyplot as plt
from sbf import visualise_fov
import spatialdata as sd
import spatialdata_plot
import seaborn as sns
import scanpy as sc
import pandas as pd
import cosmx

# =============================================================================
# Paths & Data Loading
# =============================================================================
zarr_path = "QuarterBrain.zarr"
slide = "/flatFiles"

flat_file_dir_slide = path + slide

metafile = [item for item in os.listdir(flat_file_dir_slide) if 'metadata_file' in item][0]
metafile_df = pd.read_csv(flat_file_dir_slide+ '/' + metafile)

fovfile = [item for item in os.listdir(flat_file_dir_slide) if 'fov_positions_file' in item][0]
fovfile_df = pd.read_csv(flat_file_dir_slide + '/' + fovfile)

polygon = [item for item in os.listdir(flat_file_dir_slide) if 'polygons' in item][0]
polygon_df = pd.read_csv(flat_file_dir_slide + '/' + polygon)

first_run = user_input = input("Is it the first run (0: False, 1: True): ")

if first_run == '1':
    
    fovfile_df = pd.read_csv(flat_file_dir_slide + '/' + fovfile)
    fovfile_df = fovfile_df.rename({'FOV': 'fov'}, axis = 'columns')
    fovfile_df.to_csv(path + slide + '/' + fovfile, index = False)
    
    polygon_df = pd.read_csv(flat_file_dir_slide + '/' + polygon)
    polygon_df = polygon_df.rename({'cellID': 'cell_ID'}, axis = 'columns')
    if 'cell' not in polygon_df.keys():
        polygon_df['cell'] = "c_" + polygon_df['fov'].astype(str) + '_' + polygon_df['cell_ID'].astype(str)
    polygon_df.to_csv(path + slide + '/' + polygon, index = False)
    
        
    sdata = cosmx.cosmx(path + slide, flip_image = True)
    sdata.write(zarr_path)    

# =============================================================================
# Data Loading for QC
# =============================================================================
sdata = sd.read_zarr(zarr_path)
adata = sdata.tables["table"]
print(adata.obs.keys())

# FOV positions 
# # Should be defined manually with the help of the Napari visualisation
# TMAGLACIER1
# =============================================================================
sample1 = list(range(1, 66 + 1))

samples_list = [sample1]
    
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

# ========================
# Filtering & Normalization
# ========================
print("Original dimension: ", adata.shape)
sc.pp.filter_cells(adata, min_counts = 100)
print("Dimension after filtering cells: ", adata.shape)
sc.pp.filter_genes(adata, min_cells = 1000)
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
fov =  '32'
visualise_fov(sdata, fov, coordinate = 'global')
