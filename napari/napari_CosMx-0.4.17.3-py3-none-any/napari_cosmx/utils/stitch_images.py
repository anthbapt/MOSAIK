#!/usr/bin/env python

from napari_cosmx.pairing import pair_np
from napari_cosmx.utils import _stitch as stitch
from napari_cosmx.utils._patterns import get_fov_number
from napari_cosmx.utils._stitch import fov_tqdm
from tqdm.auto import tqdm
from importlib.metadata import version
import argparse
import os
import sys
import re
import sys
import numpy as np
import pandas as pd
import tifffile
import zarr
import dask.array as da
import json

def main(args_list=None):
    parser = argparse.ArgumentParser(description='Tile CellLabels and morphology TIFFs.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--inputdir",
        help="Required: Path to CellLabels and morphology images.",
        default=".")
    parser.add_argument("--imagesdir",
        help="Optional: Path to morphology images, if different than inputdir.",
        default=None)
    parser.add_argument("-o", "--outputdir",
        help="Required: Where to create zarr output.",
        default=".")
    parser.add_argument("-f", "--offsetsdir",
        help="Required: Path to directory location containing a file ending in FOV_Locations.csv or legacy format latest.fovs.csv.",
        default=".")
    parser.add_argument("-l", "--labels",
        help="\nOptional: Only stitch labels.",
        action='store_true')
    parser.add_argument("-u", "--umperpx",
        help="Optional: Override image scale in um per pixel.\n"+
        "Instrument-specific values to use:\n-> beta04 = 0.1228",
        default=None,
        type=float)
    parser.add_argument("-z", "--zslice",
        help="Optional: Z slice to stitch.",
        default="0",
        type=int)
    parser.add_argument("--dotzarr",
        help="\nOptional: Add .zarr extension on multiscale pyramids.",
        action='store_true')
    args = parser.parse_args(args=args_list)

    # Check output directory
    if not os.path.exists(args.outputdir):
        print(f"Output path does not exist, creating {args.outputdir}")
        os.mkdir(args.outputdir)
    store = os.path.join(args.outputdir, "images")
    if not os.path.exists(store):
        os.mkdir(store)

    if args.imagesdir is None:
        args.imagesdir = args.inputdir

    # Read FOV locations file
    fov_offsets = stitch.offsets(args.offsetsdir)

    labels_res = []
    for root, dirs, files in os.walk(args.inputdir):
        labels_res += [os.path.join(root, f) for f in files 
            if re.match(r"CELLLABELS_F[0-9]+\.TIF", f.upper())]

    # Check input directory for images and get image dimensions
    im_shape = None
    if len(labels_res) == 0:
        print(f"No CellLabels_FXXX.tif files found at {args.inputdir}")
    else:
        ref_tif = labels_res[0]
        im_shape = tifffile.TiffFile(ref_tif).pages[0].shape

    ihc_res = []
    if not args.labels:
        z_string = f"_Z{args.zslice:03}" if args.zslice != 0 else "" 
        for root, dirs, files in os.walk(args.imagesdir):
            ihc_res += [os.path.join(root, f) for f in files 
                if re.match(r".*C902_P99_N99_F[0-9]+" + z_string + r"\.TIF", f.upper())]

        if len(ihc_res) == 0:
            print(f"No _FXXX{z_string}.TIF images found at {args.imagesdir}")
        else:
            ref_tif = ihc_res[0]
            with tifffile.TiffFile(ref_tif) as im:
                n = len(im.pages)
                if n <= 1:
                    sys.exit("Expecting multi-channel TIFFs")
                im_shape = im.pages[0].shape
                if im_shape is None:
                    sys.exit("No images found, exiting.")
            # get morphology kit metadata
                channels = markers = ['B','G','Y','R','U']
                tif_tags = {}
                try:
                    for tag in im.pages[0].tags.values():
                        tif_tags[tag.name] = tag.value
                    j = json.loads(tif_tags['ImageDescription'])
                    reagents = j['MorphologyKit']['MorphologyReagents']
                    mkit = {}
                    for r in reagents:
                        channel = r['Fluorophore']['ChannelId']
                        target = r['BiologicalTarget'].replace("/", "_")
                        mkit[channel] = target
                    markers = [mkit[c] for c in channels] 
                except:
                    pass # channel names left as default ['B','G','R','Y','U']
        
    fov_height = im_shape[0]
    fov_width = im_shape[1]
    dash = (fov_height/fov_width) != 1
    if args.umperpx == None:
        scale_dict = stitch.get_scales(tiff_path=ref_tif)
    else:
        scale_dict = stitch.get_scales(um_per_px=args.umperpx)
    
    top_origin_px, left_origin_px, height, width = stitch.base(
        fov_offsets, fov_height, fov_width, scale_dict, dash)
    
    if len(labels_res) != 0:
        im = da.zeros((height, width), dtype=np.uint32, chunks=stitch.CHUNKS)
        print("Stitching cell segmentation labels.")
        for fov in fov_tqdm(fov_offsets['FOV']):
            tile_path = [x for x in labels_res if get_fov_number(x) == int(fov)]
            if len(tile_path) == 0:
                tqdm.write(f"Could not find CellLabels image for FOV {fov}")
                continue
            elif len(tile_path) > 1:
                tqdm.write(f"Multiple CellLabels files found for FOV {fov}\nUsing {tile_path[0]}")
            tile = tifffile.imread(tile_path[0]).astype(np.uint32)
            pair_np(fov, tile)
            y, x = stitch.fov_origin(fov_offsets, fov, top_origin_px, left_origin_px, fov_height, scale_dict, dash)
            im[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
        
        stitch.write_pyramid(im, scale_dict, store=store, path="labels")
        #TODO: Add .zarr extension to labels if --dotzarr is used. May not be recognized by previous reader versions.
             # Needs more work before readable by napari-ome-zarr anyway

    print("Saving metadata")
    grp = zarr.open(store, mode = 'a')
    grp.attrs['CosMx'] = {
        'fov_height': fov_height,
        'fov_width': fov_width,
        'fov_offsets': fov_offsets.to_dict(),
        'scale_um': scale_dict['um_per_px'],
        'version': version('napari_cosmx')
    }

    if len(ihc_res) != 0:
        for i in range(n):
            im = da.zeros((height, width), dtype=np.uint16, chunks=stitch.CHUNKS)
            print(f"Stitching images for {markers[i]}.")
            for fov in fov_tqdm(fov_offsets['FOV']):
                tile_path = [x for x in ihc_res if get_fov_number(x) == int(fov)]
                if len(tile_path) == 0:
                    tqdm.write(f"Could not find image for FOV {fov}")
                    continue
                elif len(tile_path) > 1:
                    tqdm.write(f"Multiple image files found for FOV {fov}\nUsing {tile_path[0]}")
                with tifffile.TiffFile(tile_path[0]) as my_tiff:
                    tile = my_tiff.pages[i].asarray()
                y, x = stitch.fov_origin(fov_offsets, fov, top_origin_px, left_origin_px, fov_height, scale_dict, dash)
                im[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
            if args.dotzarr:
                markers[i] += ".zarr"
            stitch.write_pyramid(im, scale_dict, store=store, path=f"{markers[i]}")

if __name__ == '__main__':
    sys.exit(main())