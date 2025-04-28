#!/usr/bin/env python

from napari_cosmx.utils import _stitch as stitch

import argparse
import os
import sys
import numpy as np
import pandas as pd
import zarr
import dask.array as da
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Write FOV outlines to Zarr.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--inputdir",
        help="Required: Path to stitched output with targets file.",
        default=".")
    parser.add_argument("-b", "--border",
        help="Optional: Pixel width of lines.",
        default=50,
        type=int)
    args = parser.parse_args()

    store = os.path.join(args.inputdir, "images")
    if not os.path.exists(store):
        sys.exit(f"No images folder found at {args.inputdir}")
    grp = zarr.open(store, mode = 'a')

    fov_height = grp.attrs['CosMx']['fov_height']
    fov_width = grp.attrs['CosMx']['fov_width']
    um_per_px = grp.attrs['CosMx']['scale_um']
    fov_offsets = pd.DataFrame.from_dict(grp.attrs['CosMx']['fov_offsets'])
    dash = (fov_height/fov_width) != 1
    scale_dict = stitch.get_scales(um_per_px=um_per_px)
    
    top_origin_px, left_origin_px, height, width = stitch.base(
        fov_offsets, fov_height, fov_width, scale_dict, dash)
    
    border = args.border
    im = da.zeros((height, width), dtype=np.uint8)
    print("Creating FOV outlines")
    for fov in tqdm(fov_offsets['FOV']):
        agg = np.ones((fov_height, fov_width), dtype=np.uint8)
        agg[border:-border, border:-border] = 0
        y, x = stitch.fov_origin(fov_offsets, fov, top_origin_px, left_origin_px, fov_height, scale_dict, dash)
        im[y:y+agg.shape[0], x:x+agg.shape[1]] = agg
    
    stitch.write_pyramid(im, scale_dict, store=store, path="fovgrid")

if __name__ == '__main__':
    sys.exit(main())