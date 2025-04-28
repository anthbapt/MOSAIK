#!/usr/bin/env python

import dask.array as da
import numpy as np
import zarr
from skimage.segmentation import find_boundaries
from napari.utils.colormaps import AVAILABLE_COLORMAPS
import argparse

from pathlib import Path

def main(args_list=None):
    parser = argparse.ArgumentParser(description='Convert to OME-NGFF format.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--inputdir",
        help="Required: Path to stitched images directory.")
    parser.add_argument("-o", "--outputdir",
        help="Required: Where to create OME-NGFF output.")
    parser.add_argument("--channels",
        help="Optional: Which channels to initially display",
        nargs="*",
        default=None)
    args = parser.parse_args(args=args_list)

    # start from stitched output
    store = Path(args.inputdir)
    zr = zarr.open(store)

    output_path = Path(args.outputdir)
    output_path.mkdir(exist_ok=True)

    active_channels = args.channels

    channels = [item for item in zr.group_keys() if item not in ['labels', 'protein']]
    proteins = list(zr['protein'].group_keys()) if 'protein' in zr.group_keys() else []

    channels = channels + ['protein/' + i for i in proteins]
    channels.reverse() # put MM first

    attrs = dict(zr[channels[0]].attrs)
    levels = len(attrs['multiscales'][0]['datasets'])
    segmentation = [da.from_zarr(store / 'labels', component=i).map_blocks(find_boundaries) for i in range(levels)]

    stack = [da.stack(
        [da.from_zarr(store / grp, component=i) for grp in channels] + [segmentation[i]]
        ) for i in range(levels)]

    for i in range(levels):
        stack[i].to_zarr(output_path,
                        component=i,
                        overwrite=True,
                        write_empty_chunks=False,
                        dimension_separator="/")

    zrw = zarr.open(output_path, mode="r+")

    axes = attrs['multiscales'][0]['axes']
    axes = [{'name': 'c', 'type': 'channel'}] + axes

    datasets = attrs['multiscales'][0]['datasets']
    for item in datasets:
        item['coordinateTransformations'][0]['scale'] = [1.0] + item['coordinateTransformations'][0]['scale']

    zrw.attrs['multiscales'] = [{
        'axes': axes,
        'datasets': datasets,
        'name': store.parent.name,
        'version': "0.4"
        }]

    omero = [zr[grp].attrs['omero']['channels'][0] for grp in channels]
    for index, item in enumerate(omero):
        if item['color'] in AVAILABLE_COLORMAPS:
            rgba_array = AVAILABLE_COLORMAPS[item['color']].colors[-1]
            rgb_values = rgba_array[:3] * 255
            item['color'] = "{:02X}{:02X}{:02X}".format(*rgb_values.astype(int))
        else:
            item['color'] = 'FFFFFF'
        
        item['active'] = channels[index] in active_channels

    omero.append({
        "color": "00FFFF",
        "window": {"start": 0, "end": 1},
        "label": "segmentation",
        "active": True,
    })

    for item in omero:
        item["label"] = item["label"].replace("protein/", "")
    zrw.attrs['omero'] = {"channels": omero}

    labels = [da.from_zarr(store / "labels", component=i) for i in range(levels)]
    labels = [x[np.newaxis, :] for x in labels]

    labels_grp = zarr.group(store=zrw.store,
                            overwrite=True,
                            path='labels')
    labels_grp.attrs['labels'] = ["cells"]

    cells_grp = zarr.group(store=zrw.store,
                                overwrite=True,
                                path=Path('labels') / 'cells')
    multiscales = zrw.attrs['multiscales']
    multiscales[0]['name'] = "cells"
    cells_grp.attrs['multiscales'] = multiscales
    cells_grp.attrs['image-label'] = {"version": "0.4"}


    for i in range(levels):
        labels[i].to_zarr(Path(output_path) / 'labels' / 'cells',
                        component=i,
                        overwrite=True,
                        write_empty_chunks=False,
                        dimension_separator="/")
                        