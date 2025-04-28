#!/usr/bin/env python

import argparse
import zarr
import dask.array as da
import numpy as np
from scipy import ndimage
import os
import sys
import napari

def main():
    parser = argparse.ArgumentParser(description='Load zarr pyramids into Napari')
    parser.add_argument("inputdir",
        help="Path to images parent directory")
    parser.add_argument("-p", "--protein", nargs="*",
        help="Proteins to load",
        default="*")
    parser.add_argument("-s", "--scale", nargs="*",
        help="Scale of protein expression images",
        default=1,
        type=float)
    args = parser.parse_args()

    # Check input directory
    if not os.path.exists(args.inputdir):
        sys.exit(f"Input directory not found")
    if not os.path.exists(os.path.join(args.inputdir, "images")):
        sys.exit(f"No images directory found at {args.inputdir}")

    viewer = napari.Viewer()
    grp = zarr.open(os.path.join(args.inputdir, "images"), mode = 'r',)
    scale = 1
    if 'protein' in grp.group_keys():
        scale = args.scale
        if args.protein == "*":
            proteins = grp['protein'].group_keys() 
        else:
            proteins = args.protein
        for p in proteins:
            datasets = grp[f"protein/{p}"].attrs["multiscales"][0]["datasets"]
            im = [da.from_zarr(os.path.join(args.inputdir, "images", "protein", p), component=d["path"]) for d in datasets]
            viewer.add_image(im, colormap='cyan', blending="additive", name=p,
                contrast_limits = (0,1000), visible=False, rgb=False)
    channels = [i for i in grp.group_keys() if i not in ["protein", "labels"]]
    for i in channels:
        datasets = grp[f"{i}"].attrs["multiscales"][0]["datasets"]
        im = [da.from_zarr(os.path.join(args.inputdir, "images", f"{i}"), component=d["path"]) for d in datasets]
        viewer.add_image(im, colormap='gray', blending="additive", name=f"Channel {i}",
            contrast_limits = (0,2**16 - 1), scale = (scale, scale), rgb=False)
            
    if not 'labels' in grp.group_keys():
        print("No labels group found in zarr store")
    else:
        datasets = grp['labels'].attrs["multiscales"][0]["datasets"]
        kernel = np.ones((3,3))
        kernel[1, 1] = -8
        labels = [da.from_zarr(os.path.join(args.inputdir, "images", "labels"), component=d["path"]).map_blocks(
            # show edges
            lambda x: ndimage.convolve(x, kernel, output=np.uint16),
        ) for d in datasets]
        layer = viewer.add_image(labels, scale = (scale, scale), contrast_limits=(0, 1), colormap="cyan",
            blending="additive", rgb=False)
        layer.editable = False

    napari.run()

if __name__ == '__main__':
    sys.exit(main())