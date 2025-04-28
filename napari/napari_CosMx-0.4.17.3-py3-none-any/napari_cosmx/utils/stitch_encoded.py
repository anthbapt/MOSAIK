#!/usr/bin/env python

from napari_cosmx.utils import _stitch as stitch
from napari_cosmx.utils._patterns import get_fov_number
from napari_cosmx.utils._stitch import fov_tqdm
from tqdm.auto import tqdm
from pathlib import Path
import argparse
import os
import glob
import re
import sys
import numpy as np
import pandas as pd
import tifffile
import zarr
import dask.array as da

def main():
    parser = argparse.ArgumentParser(description='Tile protein expression images',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--inputdir",
        help="Required: Path to ProteinDir",
        default=".")
    parser.add_argument("-o", "--outputdir",
        help="Required: Where to create zarr output",
        default=".")
    parser.add_argument("-f", "--offsetsdir",
        help="Required: Path to directory location containing a file ending in FOV_Locations.csv or legacy format latest.fovs.csv.",
        default=".")
    parser.add_argument("-n", "--ncoder",
        help="Optional: Path to nCoder csv.",
        default=None)
    parser.add_argument("-t", "--tag",
        help="\nOptional: Tag to append to end of protein names.",
        default=None)
    parser.add_argument("-u", "--umperpx",
        help="Optional: Override image scale in um per pixel.\n"+
        "Instrument-specific values to use:\n-> beta04 = 0.1228",
        default=None,
        type=float)
    parser.add_argument("-e", "--expdir",
        help="Optional: Expression Directory under FOVxxx directory,\n"+
        "  default is 'EncodedImages'.",
        default="EncodedImages")
    parser.add_argument("-s", "--scale",
        help="Optional: Scale of encoded images relative to raw,\n"+
        "  use if encoded images are binned.",
        default=1,
        type=float)
    parser.add_argument("-c", "--cycle",
        help="Optional: Cycle to use, default is 1.",
        default=1,
        type=int)
    parser.add_argument("--dotzarr",
        help="\nOptional: Add .zarr extension on multiscale pyramids.",
        action='store_true')
    args = parser.parse_args()

    # Check output directory
    if not os.path.exists(args.outputdir):
        print(f"Output path does not exist, creating {args.outputdir}")
        os.mkdir(args.outputdir)
    store = os.path.join(args.outputdir, "images")
    if not os.path.exists(store):
        os.mkdir(store)

    fov_offsets = stitch.offsets(args.offsetsdir)
    if args.ncoder is not None:
        ncoder = pd.read_csv(args.ncoder)
        ncoder['color'] = ncoder['colorCode'].str.extract("([A-Z]{2})")
        ncoder['spot'] = ncoder.apply(lambda x: x['colorCode'].find(x['color'])+1, axis=1)
        ncoder['id'] = ncoder.apply(lambda x: f"N{x['spot']:02}_{x['color']}", axis=1)
        ncoder.index = ncoder['id']
    
    # Check directory and get protein names

    for fov in fov_offsets['FOV']:
        dirlist = [x for x in list(Path(args.inputdir).glob("FOV*")) 
                  if get_fov_number(x) == int(fov)]
        if not dirlist:
            continue
        fovdir = dirlist[0]
        tmplist = list(Path(fovdir, args.expdir).glob(f"*_C{args.cycle:03}_*_F*.TIF"))
        res = [x for x in tmplist if get_fov_number(x) == int(fov)]
        if len(res) > 0:
            break
    if not res:
        tqdm.write("No compatible TIF files found.")

    x = [re.search("^.*_(N[0-9]+)_F[0-9]+.TIF", Path(i).name) for i in res]
    proteins = [i.group(1) for i in filter(None, x)]

    if len(proteins) == 0:
        tqdm.write("Spot numbers not found in TIF files.")

    with tifffile.TiffFile(res[0]) as im:
        im_shape = im.pages[0].shape
    h = im_shape[0]
    w = im_shape[1]
    dash = (h/w) != 1
    if args.umperpx == None:
        scale_dict = stitch.get_scales(tiff_path=res[0])
    else:
        scale_dict = stitch.get_scales(um_per_px=args.umperpx)
    
    h_scale = args.scale
    w_scale = args.scale
    fov_height = h/h_scale
    fov_width = w/w_scale

    top_origin_px, left_origin_px, height, width = stitch.base(
        fov_offsets, fov_height, fov_width, scale_dict, dash)
    channel = ("BB","GG","YY","RR")
    for i in proteins:
        for c in range(4):
            name = f"{i}_{channel[c]}"
            if args.ncoder is not None:
                if name in ncoder['id']:
                    name += f"_{ncoder.loc[name,'Protein']}".replace("/", "_")
                else:
                    name += "_None"
            if args.tag is not None:
                name += f"_{args.tag}"
            if args.dotzarr:
                name += ".zarr"
            display_name = name.replace('_', ' ').replace('.zarr', '')
            im = da.zeros((height, width), dtype=np.uint16, chunks=stitch.CHUNKS)
            print(f"Stitching images for {display_name}.")
            for fov in fov_tqdm(fov_offsets['FOV']):

                dirlist = [x for x in list(Path(args.inputdir).glob("FOV*")) 
                           if get_fov_number(x) == int(fov)]
                if not dirlist:
                    tqdm.write(f"Could not find FOV directory for FOV {fov}")
                    continue
                fovdir = dirlist[0]
                if len(dirlist) > 1:
                    tqdm.write(f"Multiple FOV directories found for F{fov}\nUsing {fovdir}")
                
                tmplist = list(Path(fovdir, args.expdir).glob(f"*_C{args.cycle:03}_*_{i}_F*.TIF"))
                res = [x for x in tmplist if get_fov_number(x) == int(fov)]
                if not res:
                    tqdm.write(f"Could not find {i} encoded image for FOV {fov}")
                    continue
                tile_path = res[0]
                tile = tifffile.imread(tile_path, key = c)
                y, x = stitch.fov_origin(fov_offsets, fov, top_origin_px, left_origin_px, fov_height, scale_dict, dash)
                im[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
            
            stitch.write_pyramid(im, scale_dict, store, path=f"protein/{name}")

    print("Saving metadata")
    grp = zarr.open(os.path.join(args.outputdir, "images"), mode = 'r+')
    grp['protein'].attrs['CosMx'] = {
        'scale': args.scale
    }

if __name__ == '__main__':
    sys.exit(main())