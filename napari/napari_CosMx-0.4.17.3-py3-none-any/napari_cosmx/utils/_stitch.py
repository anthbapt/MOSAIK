from napari_cosmx import DASH_UM_PER_PX, ALPHA_UM_PER_PX, BETA_UM_PER_PX, DEFAULT_COLORMAPS

import pandas as pd
import dask.array as da
import os
import zarr
from skimage.transform import resize
import tifffile
import json
import math
from numcodecs import Zlib
from functools import partial
from tqdm.auto import tqdm as std_tqdm

zarr.storage.default_compressor = Zlib()
CHUNKS = (8192, 8192)  # 'auto' or tuple

fov_tqdm = partial(
        std_tqdm, desc='Added FOV', unit=" FOVs", ncols=40, mininterval=1.2,
        bar_format="{desc} {n_fmt}/{total_fmt}|{bar}|{percentage:3.0f}%")

def offsets(offsetsdir: str):
    """Reads FOV coordinates data.

    Args:
        offsetsdir (str): Directory location containing a file
        ending in "FOV_Locations.csv" or legacy format "latest.fovs.csv".

    Returns:
        DataFrame: coordinates of each FOV
    """
    legacy_format = True
    for filename in os.listdir(offsetsdir):
        if filename.endswith("FOV_Locations.csv"):
            print(f"Using FOV locations from {filename}")
            legacy_format = False
            df = pd.read_csv(os.path.join(offsetsdir, filename))
            #pdb.set_trace()
            df['Z_um'] = df['Z_um'] / 1e3 # convert to millimeters
            df = df.rename(columns={'Z_um':'Z_mm'})
            z_mm_index = df.columns.get_loc('Z_mm')
            df.insert(z_mm_index + 1, 'ZOffset_mm', -2.0) # hardcoded
            df.insert(z_mm_index + 2, 'ROI', 0) # hardcoded
    if legacy_format:
        print(f"Using legacy format to read FOVs")
        df = pd.read_csv(os.path.join(offsetsdir, "latest.fovs.csv"), header=None)
        cols = {k: v for k, v in enumerate(
            ["Slide", "X_mm", "Y_mm", "Z_mm", "ZOffset_mm", "ROI", "FOV", "Order"]
            )}
        df = df.rename(columns=cols)
    
    return df

def _resize(image):
    return resize(
        image,
        output_shape=(max(1, image.shape[0]//2), max(1, image.shape[1]//2)),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )

def write_pyramid(image, scale_dict, store, path):
    PYRAMID_LEVELS = math.floor(math.log2(max(image.shape)/256))
    um_per_px = scale_dict["um_per_px"]
    pyramid_scale = 1
    dimensions = ['y','x']
    datasets = [{}]*PYRAMID_LEVELS
    print(f"Writing {path} multiscale output to zarr.")
    for i in range(PYRAMID_LEVELS):
        print(f"Writing level {i+1} of {PYRAMID_LEVELS}, shape: {image.shape}, chunksize: {image.chunksize}")
        image.to_zarr(store, component=path+f"/{i}", overwrite=True, write_empty_chunks=False, dimension_separator="/")
        new_chunks = tuple([
            tuple([max(1, i//2) for i in image.chunks[0]]),
            tuple([max(1, i//2) for i in image.chunks[1]])
        ])
        if path == "composite":
            new_chunks = new_chunks + (3,)
        image = image.map_blocks(_resize, dtype=image.dtype, chunks=new_chunks)
        datasets[i] = {'path': str(i), 
                       'coordinateTransformations':[{'type':'scale', 
                       'scale':[um_per_px*pyramid_scale]*len(dimensions)}]} 
        pyramid_scale *= 2
    grp = zarr.open(store, mode = 'r+')
    grp[path].attrs['multiscales'] = [{
        'axes':[{'name': dim, 'type': 'space', 'unit': 'micrometer'} for dim in dimensions],
        'datasets': datasets,
        'type': 'resize'
        }]
    channel_name = os.path.splitext(path)[0]
    # write image intensity stats as omero metadata
    if channel_name not in ['labels', 'composite']:
        window = {}
        print("Calculating contrast limits")
        window['min'], window['max'] = int(da.min(image)), int(da.max(image))
        window['start'],window['end'] = [int(x) for x in da.percentile(image.ravel()[image.ravel()!=0], (0.1, 99.9))]
        if window['start'] - window['end'] == 0:
            if window['end'] == 0:
                print(f"\nWARNING: {channel_name} image is empty!")
                window['end'] = 1000
            else:
                window['start'] = 0
        print(f"Writing omero metadata...\n{str(window)}")
        color = DEFAULT_COLORMAPS[channel_name] if channel_name in DEFAULT_COLORMAPS else DEFAULT_COLORMAPS[None]
        grp[path].attrs['omero'] = {'name':channel_name, 'channels': [{
            'label':channel_name,
            'window': window,
            'color': color
            }]}

def base(fov_offsets, fov_height, fov_width, scale_dict, dash):
    px_per_mm = scale_dict["px_per_mm"]
    if dash:
        top_origin_px = max(fov_offsets['X_mm'])*px_per_mm + fov_height
        left_origin_px = min(fov_offsets['Y_mm'])*px_per_mm
        height = round(top_origin_px - min(fov_offsets['X_mm'])*px_per_mm)
        width = round((max(fov_offsets['Y_mm'])*px_per_mm + fov_width) - left_origin_px)
    else:
        top_origin_px = min(fov_offsets['Y_mm'])*px_per_mm - fov_height
        left_origin_px = max(fov_offsets['X_mm'])*px_per_mm
        height = round(max(fov_offsets['Y_mm'])*px_per_mm - top_origin_px)
        width = round((left_origin_px + fov_width) - min(fov_offsets['X_mm'])*px_per_mm)
    return top_origin_px, left_origin_px, height, width
 
def fov_origin(fov_offsets, fov, top_origin_px, left_origin_px, fov_height, scale_dict, dash):
    px_per_mm = scale_dict["px_per_mm"]
    if dash:
        y = round(top_origin_px - (fov_offsets[fov_offsets['FOV'] == fov].iloc[0, ]["X_mm"]*px_per_mm + fov_height))
        x = round(fov_offsets[fov_offsets['FOV'] == fov].iloc[0, ]["Y_mm"]*px_per_mm - left_origin_px)
    else:
        y = round((fov_offsets[fov_offsets['FOV'] == fov].iloc[0, ]["Y_mm"]*px_per_mm - fov_height) - top_origin_px)
        x = round(left_origin_px - fov_offsets[fov_offsets['FOV'] == fov].iloc[0, ]["X_mm"]*px_per_mm)
    return y, x

def get_scales(tiff_path=None, um_per_px=None, scale=1):
    if um_per_px is None:
        with tifffile.TiffFile(tiff_path) as im:
            try:
                tif_tags = {}
                for tag in im.pages[0].tags.values():
                    tif_tags[tag.name] = tag.value
                j = json.loads(tif_tags['ImageDescription'])
                Magnification, PixelSize_um = j['Magnification'], j['PixelSize_um']
                um_per_px = round(PixelSize_um/Magnification, 4)
                print(f"Reading pixel size and magnification from metadata... scale = {um_per_px:.4f} um/px")
            except:
                im_shape = im.pages[0].shape
                fov_height,fov_width = im_shape[0],im_shape[1]
                dash = (fov_height/fov_width) != 1
                if dash:
                    instrument = 'DASH'
                    um_per_px = DASH_UM_PER_PX
                else:
                    beta = fov_height%133 == fov_width%133 == 0
                    if beta:
                        instrument = 'BETA'
                        um_per_px = BETA_UM_PER_PX
                    else:
                        instrument = 'ALPHA'
                        um_per_px = ALPHA_UM_PER_PX
                print(f"Pixel size and magnification not found in metadata, reverting to {instrument} default: {um_per_px:.4f} um/px.")
    um_per_px = round(um_per_px/scale, 4)
    mm_per_px = um_per_px/1000
    px_per_mm = 1/mm_per_px
    if scale != 1:
        print(f"Scaling by {scale} based on user input...")
        print(f"New scale = {um_per_px:.4f} um/px")
    return {"um_per_px":um_per_px, "mm_per_px":mm_per_px, "px_per_mm":px_per_mm}  
