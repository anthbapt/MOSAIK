#!/usr/bin/env python

from napari_cosmx.utils import _stitch as stitch

import argparse
import os
import sys
import numpy as np
import pandas as pd
import zarr
import dask.array as da
import vaex
import datashader as ds
import datashader.transfer_functions as tf
from tqdm import tqdm
from numpy import random

def main():
    parser = argparse.ArgumentParser(description='Project targets to image layer.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--inputdir",
        help="Required: Path to stitched output with targets file.",
        default=".")
    parser.add_argument("-r", "--random",
        help="Optional: Assign non-zero pixels random integer from 1 to this number.",
        default=None,
        type=int)
    parser.add_argument("--genes",
        help="Optional: Stitch only specific targets",
        nargs="*",
        default=None)
    parser.add_argument("--spread",
        help="Optional: Expand targets beyond a single pixel",
        default=None,
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
    
    im = da.zeros((height, width), dtype=np.uint8)
    df = vaex.open(os.path.join(args.inputdir, "targets.hdf5"))
    if args.genes is not None:
        for x in args.genes:
            assert x in df.category_labels('target'), f"{x} not found in targets" 
    print("Creating target images")
    for fov in tqdm(fov_offsets['FOV']):
        targets = df[df['fov'] == fov].to_pandas_df()
        a = np.zeros((fov_height, fov_width))
        if args.genes is not None:
            for i, x in enumerate(args.genes):
                agg = ds.Canvas(plot_width=fov_width, plot_height=fov_height,
                                x_range=(0, fov_width), y_range=(0, fov_height)).points(
                    targets[targets['target'] == x],
                    'x', 'y'
                )
                if args.spread is not None:
                    agg = tf.spread(agg, px=args.spread)
                agg = agg.to_numpy()
                a = np.where(agg != 0, i+1, a)
        elif args.random is not None:
            rng = random.default_rng()
            targets['grp'] = rng.integers(1, args.random + 1, size=len(targets.index))
            for i in range(args.random):
                agg = ds.Canvas(plot_width=fov_width, plot_height=fov_height,
                                x_range=(0, fov_width), y_range=(0, fov_height)).points(
                    targets[targets['grp'] == i+1],
                    'x', 'y'
                )
                if args.spread is not None:
                    agg = tf.spread(agg, px=args.spread)
                agg = agg.to_numpy()
                a = np.where(agg != 0, i+1, a)
        else:
            agg = ds.Canvas(plot_width=fov_width, plot_height=fov_height,
                            x_range=(0, fov_width), y_range=(0, fov_height)).points(
                targets,
                'x', 'y'
            )
            if args.spread is not None:
                agg = tf.spread(agg, px=args.spread)
            a = agg.to_numpy()
        y, x = stitch.fov_origin(fov_offsets, fov, top_origin_px, left_origin_px, fov_height, scale_dict, dash)
        im[y:y+a.shape[0], x:x+a.shape[1]] = a
    
    stitch.write_pyramid(im, scale_dict, store=store, path="targets")

if __name__ == '__main__':
    sys.exit(main())