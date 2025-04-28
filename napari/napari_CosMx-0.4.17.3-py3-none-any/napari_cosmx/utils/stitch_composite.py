#!/usr/bin/env python

from napari_cosmx.utils import _stitch as stitch
from napari_cosmx.utils._patterns import get_fov_number
import argparse
import os
import sys
import re
import sys
import numpy as np
import pandas as pd
import zarr
import dask.array as da
from skimage import io

def main():
    parser = argparse.ArgumentParser(description='Tile CellComposite images.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--inputdir",
        help="Required: Path to CellComposite images.",
        default=".")
    parser.add_argument("-o", "--outputdir",
        help="Required: Path to existing stitched output.",
        default=".")
    parser.add_argument("-u", "--umperpx",
        help="Optional: Override image scale in um per pixel.\n"+
        "Instrument-specific values to use:\n-> beta04 = 0.1228",
        default=None,
        type=float)
    args = parser.parse_args()

    # Check output directory
    if not os.path.exists(os.path.join(args.outputdir, "images")):
        sys.exit(f"{args.outputdir}/images path does not exist.\nRun stitch-images first to create it.")
    store = os.path.join(args.outputdir, "images")
    grp = zarr.open(store, mode = 'a')

    fov_offsets = pd.DataFrame.from_dict(grp.attrs['CosMx']['fov_offsets'])
    if 'scale_um' in grp.attrs['CosMx']:
        scale_dict = stitch.get_scales(um_per_px=grp.attrs['CosMx']['scale_um'])
    elif args.umperpx is not None:
        scale_dict = stitch.get_scales(um_per_px=args.umperpx)
    else:
        sys.exit("No um_per_px in metadata or provided as argument")

    composite_res = []
    for root, dirs, files in os.walk(args.inputdir):
        composite_res += [os.path.join(root, f) for f in files 
            if re.match(r"CELLCOMPOSITE_F[0-9]*\.JPG", f.upper())]

    fov_height = grp.attrs['CosMx']['fov_height']
    fov_width = grp.attrs['CosMx']['fov_width']
    dash = (fov_height/fov_width) != 1
    
    top_origin_px, left_origin_px, height, width = stitch.base(
        fov_offsets, fov_height, fov_width, scale_dict, dash)
    
    if len(composite_res) != 0:
        im = da.zeros((height, width, 3), dtype=np.uint8, chunks=stitch.CHUNKS+ (3,))
        for fov in fov_offsets['FOV']:
            tile_path = [x for x in composite_res if get_fov_number(x) == int(fov)]
            if len(tile_path) != 1:
                print(f"Could not find CellComposite image for FOV {fov}")
                continue
            tile = io.imread(tile_path[0])
            y, x = stitch.fov_origin(fov_offsets, fov, top_origin_px, left_origin_px, fov_height, scale_dict, dash)
            im[y:y+tile.shape[0], x:x+tile.shape[1], :] = tile
            print(f"Added composite for FOV {fov}")
        
        stitch.write_pyramid(im, scale_dict, store=store, path="composite")
    else:
        print(f"No CellComposite images found at {args.inputdir}")

if __name__ == '__main__':
    sys.exit(main())