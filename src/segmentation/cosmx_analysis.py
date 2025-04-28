#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
def dummy_npwarn_decorator_factory():
  def npwarn_decorator(x):
    return x
  return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)

from py_monocle import learn_graph
from spatialdata_io import cosmx
import matplotlib.pyplot as plt
import spatialdata as sd
import spatialdata_plot
import seaborn as sns
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
import sopa


# ========================
# Paths & Data Loading
# ========================
path = '/Users/k2481276/Documents/CosMx'
slide = '/G16'
zarr_path = "G16.zarr"
os.chdir(path)

# # #Uncomment below lines if generating zarr for the first time
# fovfile = [item for item in os.listdir(path + slide) if 'fov_positions_file' in item][0]
# metafile = [item for item in os.listdir(path + slide) if 'metadata_file' in item][0]
# columns_sub = ['fov','Area','AspectRatio','CenterX_local_px','CenterY_local_px','Width',
# 'Height','Mean.PanCK','Max.PanCK','Mean.G','Max.G','Mean.Membrane','Max.Membrane',
# 'Mean.CD45','Max.CD45','Mean.DAPI','Max.DAPI','cell_ID','assay_type','version',
# 'Run_Tissue_name','Panel','cellSegmentationSetId','cellSegmentationSetName',
# 'slide_ID','CenterX_global_px','CenterY_global_px','unassignedTranscripts']

# metafile_df = pd.read_csv(path + slide + '/' + metafile)
# metafile_df = metafile_df[columns_sub]
# metafile_df.to_csv(path + slide + '/' + metafile, index = False)

# fovfile_df = pd.read_csv(path + slide + '/' + fovfile)
# fovfile_df = fovfile_df.rename({'FOV': 'fov'}, axis = 'columns')
# fovfile_df.to_csv(path + slide + '/' + fovfile, index = False)

# sdata = cosmx(path + slide)
# sdata.write(zarr_path)

sdata = sd.read_zarr(zarr_path)
adata = sdata.tables["table"]
print(adata.obs.keys())

# segmentation

def extract_image(k: int, sdata):
    image_name = str(k) + "_image"
    labels_name = str(k) + "_labels"
    points_name = str(k) + "_points"
    
    image = sdata.images[image_name]
    labels = sdata.labels[labels_name]
    points = sdata.points[points_name]
    
    sub_sdata = sd.SpatialData(images={"my_image": image},
                             labels={"my_labels": labels},
                             points={"my_points": points})
    
    img_rgb = np.transpose(image.values, (1, 2, 0))
    plt.imshow(img_rgb.astype(np.uint8))
    plt.title("Multichannel Image")
    plt.axis("off")
    plt.show()
    
    for i in range(3):
        channel_idx = i  # change to 1, 2, etc. as needed
        
        # If the image has named channels:
        if "c" in image.coords:
            channel_name = image.coords["c"].values[channel_idx]
            img_channel = image.sel(c=channel_name)
        else:
            # fallback: index directly
            img_channel = image[channel_idx]
            channel_name = f"Channel {channel_idx}"
        
        # Plot
        plt.imshow(img_channel.values, cmap="gray")
        plt.title(f"Image - {channel_name}")
        plt.axis("off")
        plt.show()
    
    return sub_sdata
    
sdata_sub = extract_image(13, sdata)

# cellpose
sopa.make_image_patches(sdata_sub, patch_width = 5000, patch_overlap=300)

sopa.segmentation.stardist(sdata_sub, model_type='2D_versatile_he', key_added = "stardist")


sopa.settings.parallelization_backend = 'dask'
sopa.segmentation.cellpose(sdata_sub, ["b"], diameter=300, key_added = "cellpose_300_b")


sdata_sub.pl.render_images("my_image").pl.render_shapes(
    "cellpose_300_b", fill_alpha=0, outline_color="#fff", outline_alpha=1
).pl.show("global")

sopa.segmentation.combine(sdata_sub, ["cellpose_400_b", "cellpose_300_b"], key_added="combined_cells", threshold=0.2)
# remove small area

# # cellpose
# sopa.make_image_patches(sdata_1, patch_width = 5000)
# sopa.settings.parallelization_backend = 'dask'
# sopa.segmentation.cellpose(sdata_1, ["r", "b"], diameter=400, key_added = "cellpose")

# sdata_1.pl.render_images("my_image").pl.render_shapes(
#     "combined_cells", fill_alpha=0, outline_color="#fff", outline_alpha=1
# ).pl.show("global")

# sopa.segmentation.cellpose(sdata_1, ["r", "b"], diameter=200, key_added = "combined_200")

# sopa.segmentation.combine(sdata_1, ["cellpose", "cellpose_200"], key_added="combined_cells", threshold=0.2)




# comseg
sopa.settings.parallelization_backend = 'dask'
sopa.make_transcript_patches(sdata_sub, patch_width = 1500, write_cells_centroids=True, prior_shapes_key="cellpose_300_b")
config = {
    "dict_scale": {"x": 1, "y": 1, "z": 1},  # spot coordinates already in µm
    "mean_cell_diameter": 20,
    "max_cell_radius": 40,
    "norm_vector": False,
    "alpha": 0.5,  # alpha value to compute the polygon https://pypi.org/project/alphashape/
    "allow_disconnected_polygon": False,
    "min_rna_per_cell": 500,  # minimal number of RNAs for a cell to be taken into account
    "gene_column": "target",
}

sopa.segmentation.comseg(sdata_sub, config=config, min_area=50, key_added = "comseg")
sdata_sub.pl.render_images("my_image").pl.render_shapes(
     "comseg", fill_alpha=0, outline_color="#fff", outline_alpha=1
 ).pl.show("global")                        






sopa.segmentation.baysor(sdata_sub, scale=2, min_area=10, key_added = "baysor")






###############################################################################
#Import single cell atlas for cell typing
###############################################################################
# Cardiac tissue, cardiomyocytes
# AVN_aCMs_lognormalised.h5ad (34MB)
# SAN_aCMs_lognormalised.h5ad (247MB)
# human_dcm_hcm_scportal_03.17.2022.h5ad (13.5GB)
# Global_lognormalised.h5ad (8.7GB)
atlas = sc.read_h5ad("/Users/k2481276/Documents/CosMx/G16_scatlas/Global_lognormalised.h5ad")
sc.tl.pca(atlas)
sc.pp.neighbors(atlas, method = "umap", n_pcs = 10)
sc.pl.scatter(atlas, basis = "umap", color = ["cell_state"], components = [1, 2])




# Calculate quality control metrics
adata.var["NegPrb"] = adata.var_names.str.startswith("NegPrb")
sc.pp.calculate_qc_metrics(adata, qc_vars=["NegPrb"], inplace=True)
pd.set_option("display.max_columns", None)
adata.obs["total_counts_NegPrb"].sum() / adata.obs["total_counts"].sum() * 100

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].set_title("Total transcripts per cell")
sns.histplot(
    adata.obs["total_counts"],
    kde=False,
    ax=axs[0])

axs[1].set_title("Unique transcripts per cell")
sns.histplot(
    adata.obs["n_genes_by_counts"],
    kde=False,
    ax=axs[1])

axs[2].set_title("Transcripts per FOV")
sns.histplot(
    adata.obs.groupby("fov")["total_counts"].sum(),
    kde=False,
    ax=axs[2])
plt.savefig('statistics.png', format = 'png', dpi = 600)



sc.pp.filter_cells(adata, min_counts = 100)
sc.pp.filter_genes(adata, min_cells = 400)

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "leiden"], wspace=0.4, save=True)

sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=["leiden"], 
                      wspace=0.4, save="leiden.png", dpi=600)


sdata.pl.render_images().pl.show(coordinate_systems=["2"])
sdata.pl.render_labels().pl.show(coordinate_systems=["2"])
sdata.pl.render_points().pl.show(coordinate_systems=["2"])

sdata.points["2_points"] = sdata.points["2_points"].sample(frac=0.01)
sdata.pl.render_points().pl.show(coordinate_systems=["2"])
