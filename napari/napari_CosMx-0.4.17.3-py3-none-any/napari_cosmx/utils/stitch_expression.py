#!/usr/bin/env python

from napari_cosmx.utils import _stitch as stitch
from napari_cosmx.utils._patterns import get_fov_number, convertLabels
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
from pathlib import Path


def makeConfig(args) -> dict:
    """Checking input paths

    Args:
        args: args parser arguments

    Returns:
        dict: configuration dictionary
    """

    inputdir = args.inputdir

    # Inputdir exists
    if not os.path.isdir(inputdir):
        sys.exit("Input direcotry not found. Check spelling.")
    
    # Plex file exists
    plexFile = None
    for filename in os.listdir(inputdir):
        if filename.startswith("plex"):
            plexFile = os.path.join(inputdir, filename)
    if plexFile is None:
        sys.exit("No plex file found. Aborting.")

    # ProteinDir exists
    ProteinDir = os.path.join(inputdir, "ProteinDir")
    if not os.path.exists(ProteinDir):
        sys.exit(f"Cound not find {str(ProteinDir)}.")

    # AnalysisResults and subdir exist
    AnalysisDirParent = os.path.join(inputdir, 'AnalysisResults')
    if not os.path.exists(AnalysisDirParent):
        sys.exit(f"Cound not find {str(AnalysisDirParent)}.")
    AnalysisDirSubBasename = [i for i in os.listdir(AnalysisDirParent) if not i.startswith('.')]
    AnalysisDir = os.path.join(AnalysisDirParent, AnalysisDirSubBasename[0])
    if not os.path.exists(AnalysisDir):
        sys.exit(f"Cound not find {str(AnalysisDir)}.")   

    # CellStatsDir exists
    CellStatsDir = os.path.join(inputdir, "CellStatsDir")
    if not os.path.exists(CellStatsDir):
        sys.exit(f"Cound not find {str(CellStatsDir)}.")
    
    # RunSummary exists
    RunSummaryDir = os.path.join(inputdir, "RunSummary")
    if not os.path.exists(RunSummaryDir):
        sys.exit(f"Cound not find {str(RunSummaryDir)}.")
    
    # location of ProteinImages directory(ies)
    check = [os.path.exists(os.path.join(AnalysisDir, x, "ProteinImages")) for x in os.listdir(AnalysisDir)]
    if all(check):
        # ProteinImages is found in AnalysisResults/abc..j/FOV.../
        protein_images_in_analysis_results = True
    elif any(check):
        sys.exit("Could not find some of the ProteinImages folders within the AnalysisResults")
    else: 
        # Look instead within the ProteinDir (legacy format)
        protein_images_in_analysis_results = False

    # 3 or 5 digit format
    fov_dirs = [item for item in Path(CellStatsDir).iterdir() if item.is_dir() and item.name.startswith("FOV")]
    query = str(fov_dirs[0])

    f_length = len(query.split("/")[-1].split("FOV", 1)[1])

    # FOV offsets data frame
    fov_offsets = stitch.offsets(RunSummaryDir)

    # Protein names from plex file.
    proteins_df = pd.read_csv(plexFile, sep="\t", usecols=["ProbeID", "DisplayName"])
    proteins = dict(zip(proteins_df['ProbeID'], proteins_df['DisplayName']))
    if len(proteins) == 0:
        sys.exit("Could not convert plex file.")
    
    # Get shape information from first FOV
    fov = str(fov_offsets['FOV'][0]).zfill(f_length)
    if protein_images_in_analysis_results:
        FOVDir = os.path.join(AnalysisDir, "FOV" + fov)
    else:
        # legacy location inside ProteinDir
        FOVDir = os.path.join(ProteinDir, "FOV" + fov)
    FOVProteinImagesDir = os.path.join(FOVDir, "ProteinImages")
    
    prototype_file = None
    for item in Path(FOVProteinImagesDir).iterdir():
        if item.is_file() and item.name.endswith("TIF"):
            prototype_file = os.path.join(FOVProteinImagesDir, item.name)
    if prototype_file is None:
        sys.exit("Could not find TIF files.")


    # Extract image information from first image
    with tifffile.TiffFile(prototype_file) as im:
        im_shape = im.pages[0].shape
    h = im_shape[0]
    w = im_shape[1]
    dash = (h/w) != 1
    if args.umperpx == None:
        scale_dict = stitch.get_scales(tiff_path=prototype_file)
    else:
        scale_dict = stitch.get_scales(um_per_px=args.umperpx)

    h_scale = args.scale
    w_scale = args.scale
    fov_height = h/h_scale
    fov_width = w/w_scale

    top_origin_px, left_origin_px, height, width = stitch.base(
        fov_offsets, fov_height, fov_width, scale_dict, dash) 

    return({'ProteinDir':ProteinDir, 'AnalysisDir':AnalysisDir, 'CellStatsDir':CellStatsDir, 
            'RunSummaryDir':RunSummaryDir, 'protein_images_in_analysis_results':protein_images_in_analysis_results,
            'f_length':f_length, 'fov_offsets':fov_offsets, 'proteins':proteins, 'top_origin_px':top_origin_px,
            'left_origin_px':left_origin_px, 'fov_height':fov_height, 'height':height, 'fov_width':fov_width, 'width':width,
            'scale_dict':scale_dict, 'dash':dash})

def main():
    parser = argparse.ArgumentParser(description='Tile protein expression images',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--inputdir",
        help="Required: Path to parent directory containing CellStatsDir, RunSummary, AnalysisResults, and ProteinDir.",
        default=".")
    parser.add_argument("-o", "--outputdir",
        help="Required: Where to create zarr output.",
        default=".")
    parser.add_argument("-t", "--tag",
        help="\nOptional: Tag to append to end of protein names.",
        default=None)
    parser.add_argument("-u", "--umperpx",
        help="Optional: Override image scale in um per pixel.\n"+
        "Instrument-specific values to use:\n-> beta04 = 0.1228",
        default=None,
        type=float)
    parser.add_argument("-s", "--scale",
        help="Optional: Scale of expression images relative to raw,\n"+
        "  use if decoded images are binned.",
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

    # Gather configuratioins
    config = makeConfig(args)
    fov_offsets = config['fov_offsets']
    
    for i in config['proteins']:
        im = da.zeros((config['height'], config['width']), dtype=np.uint16, chunks=stitch.CHUNKS)
        if "CPRO" in i:
            is_cpro = True
            print(f"Stitching images for protein {i} ({convertLabels(i, config['proteins'])}).")
        else:
            is_cpro = False
            print(f"Stitching images for protein {i}).")

        for fov in fov_tqdm(fov_offsets['FOV']):
            if config["protein_images_in_analysis_results"]:
                dirlist = [x for x in list(Path(config["AnalysisDir"]).glob("FOV*")) 
                        if get_fov_number(x) == int(fov)]
            else:
                dirlist = [x for x in list(Path(config["ProteinDir"]).glob("FOV*")) 
                        if get_fov_number(x) == int(fov)]
            if not dirlist:
                tqdm.write(f"Could not find FOV directory for FOV {fov}")
                continue
            fovdir = dirlist[0]
            if len(dirlist) > 1:
                tqdm.write(f"Multiple FOV directories found for F{fov}\nUsing {fovdir}")
            
            fov_tiff_files_list = list(Path(fovdir, "ProteinImages").glob(f"*_C{args.cycle:03}_F*_{i}.TIF"))
            tile_path = [x for x in fov_tiff_files_list if get_fov_number(x) == int(fov)]
            if not tile_path:
                tqdm.write(f"Could not find {i} protein image for FOV {fov}")
                continue
            tile_path = tile_path[0]
            tile = tifffile.imread(tile_path)
            y, x = stitch.fov_origin(fov_offsets, fov, config['top_origin_px'], config['left_origin_px'], config['fov_height'], config['scale_dict'], config['dash'])
            im[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
        if args.tag:
            i += f"_{args.tag}"
        if args.dotzarr:
            i += ".zarr"
        if is_cpro:
            stitch.write_pyramid(im, config['scale_dict'], store, path=f"protein/{convertLabels(i, config['proteins'])}")
        else:
            stitch.write_pyramid(im, config['scale_dict'], store, path=f"protein/{i}")

    print("Saving metadata")
    grp = zarr.open(os.path.join(args.outputdir, "images"), mode = 'r+')
    grp['protein'].attrs['CosMx'] = {
        'scale': args.scale
    }

if __name__ == '__main__':
    sys.exit(main())
