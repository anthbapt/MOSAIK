#!/usr/bin/env python

import numpy as np
from tifffile import TiffWriter, imread
import zarr
import dask.array as da
import argparse
import os
import sys
from scipy import ndimage
from tqdm import tqdm
from skimage.transform import resize
from pathlib import Path
from dask.diagnostics import ProgressBar
import tempfile
from napari_cosmx.pairing import pair
import pandas as pd
import re

def split_list_element(input_list):
    """
    Splits a list element by spaces or commas.

    Args:
        input_list: A list of length 1 containing a string.

    Returns:
        A list of strings, or the original list if the input is invalid.
        Returns an empty list if the input list is empty.
    """

    if not isinstance(input_list, list):
        return "Input must be a list"  # Or raise an exception

    if not input_list:
        return [] #Handle empty list

    if len(input_list) != 1:
        return "Input list must have length 1"  # Or raise an exception

    element = input_list[0]

    if "Cathepsin B" in element:
        element = element.replace("Cathepsin B", "CATHEPSIN_B_PLACEHOLDER") # Use a placeholder
        split_elements = re.split(r'[ ,]+', element)
        cleaned_elements = [item.replace("CATHEPSIN_B_PLACEHOLDER", "Cathepsin B") if item == "CATHEPSIN_B_PLACEHOLDER" else item for item in split_elements if item]
    else:
        split_elements = re.split(r'[ ,]+', element)
        cleaned_elements = [item for item in split_elements if item]

    return cleaned_elements

class BatchStorage:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.storage = {}
        self.current_key = None
        self.item_count = 0
        self.labels = None  # Store labels separately

    def set_labels(self, labels):
        """Sets the labels to be added to each batch."""
        self.labels = labels

    def add_item(self, item, new_batch=False):
        """Adds an item to the storage, including handling labels."""

        if not self.storage:
            self.current_key = 0
            self.storage[self.current_key] = []
            self.item_count = 0
            if self.labels is not None:  # Add labels to the first batch if available
                self.storage[self.current_key].append(self.labels)

        if new_batch or self.item_count >= self.batch_size:
            self.current_key = max(self.storage.keys()) + 1 if self.storage else 0
            self.storage[self.current_key] = []
            self.item_count = 0
            if self.labels is not None: # Add labels to the new batch if available
                self.storage[self.current_key].append(self.labels)

        self.storage[self.current_key].append(item)
        self.item_count += 1

    def get_batch(self, key):
        return self.storage.get(key)

    def __str__(self):
        return str(self.storage)

def _edges(x):
    kernel = np.ones((3,3))
    kernel[1, 1] = -8
    arr = ndimage.convolve(x, kernel, output=np.int32) #changed to int32 to handle negative values
    arr[arr != 0] = 1 #still binarize, but now it is 0 and 1
    return arr.astype('uint8')

def _scale_edges(x):
    # Scale the binary [0, 1] to [100, 10000]
    x_scaled = x * 9900 + 100  # Linear scaling
    return x_scaled.astype('uint16') #return as uint16, as it is larger than uint8

def Parse():
    parser = argparse.ArgumentParser(description='Export stitched Zarr to OME-TIFF')
    parser.add_argument("-i", "--inputdir",
        help="Required: Path to existing stitched output.",
        default=".")
    parser.add_argument("-o", "--outputdir",
        help="Required: Path to write OME-TIFF file.",
        default=".")
    parser.add_argument("--filename",
        help="Name for OME-TIFF file, use ome.tif extension.",
        default="cosmx-wsi.ome.tif")
    parser.add_argument("--compression",
        help="Passed to TiffWriter, default is 'zlib'. "+
        "Other options include 'lzma' (smallest), 'lzw', and 'none'",
        default='zlib')
    parser.add_argument("-b", "--batchsize",
        help="Required: the number of elements to put into each ome-tiff file. Recommended = 5 or fewer.\n",
        default=5,
        type=int)
    parser.add_argument("-s", "--segmentation",
        help="\nOptional: Create TIFF for segmentation mask.",
        action='store_true')
    parser.add_argument("-c", "--channels",
        help="Optional: Output only specific morphology channels",
        nargs="*",
        default=None)
    parser.add_argument("-p", "--proteins",
        help="Optional: Output only specific proteins",
        nargs="*",
        default=None)
    parser.add_argument("--levels",
        help="Optional: Specify number of pyramid levels.\n",
        default=8,
        type=int)
    parser.add_argument("-v", "--verbose",
        help="Print verbose output?",
        action='store_true')
    parser.add_argument("--libvips",
        help="\nOptional: Use libvips to create pyramidal image, will be slower but more memory-efficient.",
        action='store_true')
    parser.add_argument("--vipshome",
        help="Optional: Path to vips binaries. Required in Windows if vips and associated DLLs are not in PATH",
        default=None)
    parser.add_argument("--vipsconcurrency",
        help="Optional: Specify number of threads for vips.\n",
        default=8,
        type=int)
    
    args = parser.parse_args()

    return(args)

def main():
    
    parser = argparse.ArgumentParser(description='Export stitched Zarr to OME-TIFF')
    parser.add_argument("-i", "--inputdir",
        help="Required: Path to existing stitched output.",
        default=".")
    parser.add_argument("-o", "--outputdir",
        help="Required: Path to write OME-TIFF file.",
        default=".")
    parser.add_argument("--filename",
        help="Name for OME-TIFF file, use ome.tif extension.",
        default="cosmx-wsi.ome.tif")
    parser.add_argument("--compression",
        help="Passed to TiffWriter, default is 'zlib'. "+
        "Other options include 'lzma' (smallest), 'lzw', and 'none'",
        default='zlib')
    parser.add_argument("-b", "--batchsize",
        help="Required: the number of elements to put into each ome-tiff file. Recommended = 5 or fewer.\n",
        default=5,
        type=int)
    parser.add_argument("-s", "--segmentation",
        help="\nOptional: Create TIFF for segmentation mask.",
        action='store_true')
    parser.add_argument("-c", "--channels",
        help="Optional: Output only specific morphology channels",
        nargs="*",
        default=None)
    parser.add_argument("-p", "--proteins",
        help="Optional: Output only specific proteins",
        nargs="*",
        default=None)
    parser.add_argument("--levels",
        help="Optional: Specify number of pyramid levels.\n",
        default=8,
        type=int)
    parser.add_argument("-v", "--verbose",
        help="Print verbose output?",
        action='store_true')
    parser.add_argument("--libvips",
        help="\nOptional: Use libvips to create pyramidal image, will be slower but more memory-efficient.",
        action='store_true')
    parser.add_argument("--vipshome",
        help="Optional: Path to vips binaries. Required in Windows if vips and associated DLLs are not in PATH",
        default=None)
    parser.add_argument("--vipsconcurrency",
        help="Optional: Specify number of threads for vips.\n",
        default=8,
        type=int)
    
    args = parser.parse_args()

    # Check output directory
    if not os.path.exists(args.outputdir):
        print(f"Output path does not exist, creating {args.outputdir}")
        os.mkdir(args.outputdir)
    store = os.path.join(args.inputdir, "images")
    if not os.path.exists(store):
        sys.exit(f"Could not find images directory at {args.inputdir}")

    # Top-level input, checking, initializing
    zarr_array = zarr.open(store, mode='r')
    if 'scale_um' in zarr_array.attrs['CosMx']:
        pixelsize = zarr_array.attrs['CosMx']['scale_um']
    else:
        sys.exit("Could not find scaling information from top-level zarr. Error 1.")
    

    batches = BatchStorage(args.batchsize) # intialize

    ### Segmentation
    has_labels = False
    idx = 0
    if args.segmentation:
        for item in zarr_array.items():
            if item[0] == "labels":
                has_labels = True
                idx = 1
        if not has_labels:
            sys.exit(f"Error. Segmentation labels were requested but the directory 'labels' could not be found.")
        else:
            if args.verbose:
                print(f"Adding segmentation to each batch.")
            batches.set_labels('labels')

    ### Channels
    if args.channels is not None:
        valid_channels = [key for key in zarr_array.keys() if key not in ["labels", "protein"]]
        if len(args.channels) == 0:
            channels_to_process = valid_channels
        else:
            cleaned_list = split_list_element(args.channels)
            channels_to_process = []
            for x in cleaned_list:
                if x in valid_channels:
                    channels_to_process.append(x)
                else:
                    print(f"Warning! {x} is not a valid channel and will be ignored.")
        if len(channels_to_process) == 0:
            print(f"--channels were requested but no valid channels were found in the zarr store.") 
        else:
            [batches.add_item(x) for x in channels_to_process]
    
    ### Proteins
    has_proteins = False
    for item in zarr_array.items():
        if item[0] == "protein":
            has_proteins = True

    if args.proteins is not None and has_proteins:
        valid_proteins = ['protein/' + x for x in zarr_array['protein'].group_keys()]
        if len(args.proteins) == 0:
            # requests all proteins
            proetins_to_process = valid_proteins
        else:
            cleaned_list = ['protein/' + x for x in split_list_element(args.proteins)]
            proetins_to_process = []
            for x in cleaned_list:
                if x in valid_proteins:
                    proetins_to_process.append(x)
                else:
                    print(f"Warning! {x} is not a valid protein and will be ignored (check spelling).")
        if len(proetins_to_process) == 0:
            print(f"--proteins were specified but no valid protein names were given.")
        else:
            [batches.add_item(x) for x in proetins_to_process]
    elif args.proteins and args.verbose:
        print(f"Requested protein export but no protein zarr found. Ignoring.")

    ### Processing
    for key, items in batches.storage.items():
        if args.verbose:
            print(f"Processing batch number {str(key)}")
        # get attributes from first item (skip labels if applicable)
        attrs = zarr_array[items[idx]].attrs
        datasets = attrs["multiscales"][0]["datasets"] # dimensions of each level
        omero = attrs["omero"]
        window = omero['channels'][0]['window']
        names = [x.replace(".zarr", "").replace("protein/", "") for x in items]
        
        pyramid_levels = len(datasets)
        if args.levels is not None:
            pyramid_levels = args.levels
        
        item_metadata={
            'axes': 'CYX', # single channel for now
            'Channel': {'Name': names},
            'PhysicalSizeX': pixelsize,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': pixelsize,
            'PhysicalSizeYUnit': 'µm',
            'ContrastLimits': [window['min'], window['max']], # for now, restricted to first item
            'Window': {'Start': window['start'], 'End': window['end']}
        }

        levels = []
        for d in datasets:
            arrays = []
            for i in items:
                i_array = da.from_zarr(store + f"/{i}", component=d["path"])
                if has_labels and i == "labels":
                    i_array = i_array.map_blocks(_edges).map_blocks(_scale_edges) # binary mask, bounded
                arrays.append(i_array)
            stacked_array = da.stack(arrays)
            levels.append(stacked_array)

        data = levels[0]
        resolution = tuple(datasets[0]['coordinateTransformations'][0]['scale'])
        path = os.path.join(args.outputdir, f"batch_{key}_{args.filename}")
        if args.verbose:
            print(f"Writing {path}.")

        if not args.libvips:
            # Needs lots of RAM
            with TiffWriter(path, bigtiff=True, ome=True) as tif:
                #print(f"Writing pyramid levels for batch {key}.")
                options = dict(
                    resolutionunit='MICROMETER',
                    tile=(1024, 1024),
                    metadata=item_metadata,
                    subifds=pyramid_levels - 1,
                    compression=args.compression
                )
                for i in tqdm(range(pyramid_levels), ncols=60, smoothing=1):
                    tif.write(
                        data=data,
                        resolution=resolution,
                        **options
                    )
                    if i == 0:
                        del options['metadata']
                        del options['subifds']
                    if i < len(datasets) - 1:
                        data = levels[i + 1]
                        resolution = tuple(datasets[i + 1]['coordinateTransformations'][0]['scale'])
                    else:
                        data = resize(
                            data,
                            output_shape=(data.shape[0], data.shape[1] // 2, data.shape[2] // 2),
                            order=0,
                            preserve_range=True,
                            anti_aliasing=False
                        )
                        resolution = tuple(2*i for i in resolution)
        else:
            if os.name == "nt" and args.vipshome is not None:
                os.environ['PATH'] = args.vipshome + ';' + os.environ['PATH']
            os.environ['VIPS_PROGRESS'] = "1"
            os.environ['VIPS_CONCURRENCY'] = str(args.vipsconcurrency)
            import pyvips # must import after updating path
            tmpdirname = tempfile.mkdtemp(dir=args.outputdir)
            tmptif = os.path.join(tmpdirname, 'tmp.ome.tif')
            with TiffWriter(tmptif, bigtiff=True, ome=True) as tif:
                print(f"Writing uncompressed empty tiff for {str(key)}, shape: {data.shape}.\nThis could take a while.")
                options = dict(
                    resolutionunit='MICROMETER',
                    tile=(1024, 1024),
                    metadata=item_metadata,
                    compression=None,
                )
                tif.write(
                    data=None,
                    dtype=data.dtype,
                    shape=data.shape,
                    resolution=resolution,
                    **options
                )
            # Open TIFF as Zarr
            store = imread(tmptif, mode='r+', aszarr=True)
            z = zarr.open(store, mode='r+')
            if data.shape[0] == 1 and len(data.shape) ==3:
                print(f"Only 1 layer for this ome-tiff file. Reshaping.")
                # must reshape the dask array in this edge case
                data = data.squeeze(axis=0)
            print(f"Writing data to tiff as zarr")
            with ProgressBar():
                da.to_zarr(arr=data, url=z)
            store.close()

            print(f"Creating pyramidal OME-TIFF at {path}")
            image = pyvips.Image.new_from_file(
                tmptif,
                n = len(items)
            )
            tile_height = data.shape[-2] // 2**(pyramid_levels - 1) # 3D
            tile_width = data.shape[-1] // 2**(pyramid_levels - 1) # 3D
            tile_height = tile_height + 16 - (tile_height % 16)
            tile_width = tile_width + 16 - (tile_width % 16)
            image.tiffsave(
                path,
                compression="deflate", bigtiff=True, 
                tile=True, tile_width=tile_width, tile_height=tile_height,
                pyramid=True, subifd=True
            )

if __name__ == '__main__':
    sys.exit(main())