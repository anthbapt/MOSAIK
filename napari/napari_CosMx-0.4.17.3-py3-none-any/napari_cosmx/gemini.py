import napari
from napari.experimental import link_layers
from napari_cosmx import DASH_MM_PER_PX, ALPHA_MM_PER_PX, BETA_MM_PER_PX, DEFAULT_COLORMAPS, OTHER_KEYS
from superqt.utils import qdebounced
from skimage import io
from skimage.segmentation import find_boundaries
from scipy import ndimage
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os
import warnings
import re
import glob
import anndata as ad
import zarr
import vaex
import dask.array as da
import pickle
from napari_cosmx.pairing import unpair, pair
from napari_cosmx._dock_widget import GeminiQWidget
from sklearn import preprocessing
from skimage.draw import polygon, ellipse
from scipy import ndimage as ndi
from napari.utils.colormaps import AVAILABLE_COLORMAPS, label_colormap, color_dict_to_colormap
from napari.utils.colormaps.standardize_color import transform_color
from vispy.color.colormap import Colormap
from inspect import signature
from os import listdir
from importlib.metadata import version

class Gemini:
    """Initialize new instance and launch viewer

    Args:
        path (str): path to adata, transcripts, and/or images or .h5ad path
        viewer (napari.viewer.Viewer): If None, napari will be launched.
    """
    def __init__(self, path, viewer=None):
        assert os.path.exists(path), f"Could not find {path}"
        self._rotate = 0
        self.folder = path
        self.adata = None
        self.fov_labels = None
        self.segmentation_layer = None
        self.cells_layer = None
        self.color_cells_layer = None
        if path.endswith(".h5ad"):
            self.adata = ad.read(path)
            self.folder = os.path.dirname(path)
        files = next(os.walk(self.folder))[2]
        if self.adata is None:
            res = [f for f in files if f.endswith('.h5ad')]
            if len(res) != 1:
                print("Could not find AnnData .h5ad file")
            else:
                self.adata = ad.read(os.path.join(self.folder, res[0]))
        assert os.path.exists(os.path.join(self.folder, "images")), f"No images directory found at {self.folder}"
        self.grp = zarr.open(os.path.join(self.folder, "images"), mode = 'r+',)
        self.is_protein = 'protein' in self.grp.group_keys()
        self.fov_height = self.grp.attrs['CosMx']['fov_height']
        self.fov_width = self.grp.attrs['CosMx']['fov_width']
        self.fov_offsets = pd.DataFrame.from_dict(self.grp.attrs['CosMx']['fov_offsets'])
        self.dash = (self.fov_height/self.fov_width) != 1
        self.beta = self.fov_height%133 == self.fov_width%133 == 0
        # try to get scale from CosMx metadata, otherwise revert to instrument defaults
        if 'scale_um' in self.grp.attrs['CosMx']:
            um_per_px = self.grp.attrs['CosMx']['scale_um']
            print(f"Reading CosMx scale from .attrs: {um_per_px:.4f} um/px")
            self.mm_per_px = um_per_px/1000
        elif 'scale_mm' in self.grp.attrs['CosMx']:
            self.mm_per_px = self.grp.attrs['CosMx']['scale_mm']
            print(f"Reading CosMx scale from .attrs: {self.mm_per_px} mm/px")
        else:
            if self.dash:
                self.mm_per_px = DASH_MM_PER_PX
                self.instrument = 'DASH'
            elif self.beta:
                self.mm_per_px = BETA_MM_PER_PX
                self.instrument = 'BETA'
            else:
                self.mm_per_px = ALPHA_MM_PER_PX
                self.instrument = 'ALPHA'
            print(f"No CosMx scale found in .attrs, using {self.instrument} default: {(self.mm_per_px*1000):.4f} um/px")
        
        self._channels = sorted(set([s.replace(r'.zarr', '') for s in self.grp.group_keys()]) - set(OTHER_KEYS))
        self.targets = None
        self._proteins = []
        self.expr_scale = 1
        if not self.is_protein:
            if not os.path.exists(os.path.join(self.folder, "targets.hdf5")):
                print(f"No targets.hdf5 file found at {self.folder}")    
            else:
                df = vaex.open(os.path.join(self.folder, "targets.hdf5"))
                offsets = vaex.from_pandas(self.fov_offsets.loc[:, ['X_mm', 'Y_mm', 'FOV']])
                self.targets = df.join(offsets, left_on='fov', right_on='FOV')
                if self.dash:
                    self.targets['global_x'] = self.targets.x + self.targets.Y_mm*(1/self.mm_per_px) - self._top_left_mm()[1]*(1/self.mm_per_px)
                    self.targets['global_y'] = self.targets.y - self.targets.X_mm*(1/self.mm_per_px) - self._top_left_mm()[0]*(1/self.mm_per_px)
                else:
                    self.targets['global_x'] = self.targets.x - self.targets.X_mm*(1/self.mm_per_px) - self._top_left_mm()[1]*(1/self.mm_per_px)
                    self.targets['global_y'] = self.targets.y + self.targets.Y_mm*(1/self.mm_per_px) - self._top_left_mm()[0]*(1/self.mm_per_px)
        else:
            self._proteins = [s.replace(r'.zarr', '') for s in self.grp['protein'].group_keys()]
            if 'CosMx' in self.grp['protein'].attrs:
                self.expr_scale = self.grp['protein'].attrs['CosMx']['scale']
        self.name = self.adata.uns['name'] if (self.adata is not None) and ('name' in self.adata.uns) else \
            os.path.basename(self.folder)

        if self.adata is None:
            # search for *_metadata.csv
            res = glob.glob(os.path.join(self.folder, "*_metadata.csv"))
            if len(res) == 1:
                self.read_metadata(res[0])
            else:
                self.metadata = None
                print(f"No AnnData .h5ad or _metadata.csv file found at {self.folder}")
        else:
            if self.adata.obs.index.name == "cell_ID":
                self.adata.obs['cell_ID'] = self.adata.obs.index
                res = [re.search("c_.*_(.*)_(.*)", i) for i in self.adata.obs.index]
                self.adata.obs['UID'] = [pair(int(i.group(1)), int(i.group(2))) for i in res]
                self.adata.obs.set_index('UID', inplace=True)
                self.metadata = self.adata.obs
            elif self.adata.obs.index.name != "UID":
                print(f"Expected index cell_ID or UID in AnnData obs, not {self.adata.obs.index}, \
                    unable to read metadata")
            else:
                self.metadata = self.adata.obs

        # launch viewer
        if viewer:
            self.viewer = viewer
        else:
            self.viewer = napari.Viewer()
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.update_console()
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "mm"
        if any("labels" in x for x in self.grp.group_keys()):
            self.add_cell_labels()
            self.add_segmentation()
        self.add_fov_labels()
    
    def __repr__(self):
        return f'napari_cosmx.gemini.Gemini(r"{self.folder}")'

    def show_widget(self):
        if self.viewer is not None:
            widgets = [w for w in self.viewer.window._dock_widgets.values()
                if isinstance(w.widget(), GeminiQWidget)]
            if len(widgets) != 0:
                for w in widgets:
                    w.show()
            else:
                self.viewer.window.add_dock_widget(GeminiQWidget(self.viewer, self),
                    area='right',
                    name=self.name)

    def _top_left_mm(self):
        """Return (y, x) tuple of mm translation for image layers
        """
        if self.dash:
            return (-max(self.fov_offsets['X_mm']), min(self.fov_offsets['Y_mm']))
        else:
            return (min(self.fov_offsets['Y_mm']), -max(self.fov_offsets['X_mm']))

    def omero(self, channel_name, protein=False, auto=True):
        """Return omero metadata
        https://ngff.openmicroscopy.org/latest/#omero-md
            - contrast limits for image as tuple
            - colormap name

        Args:
            channel_name (str): Name of channel (e.g. "DAPI" or "4-1BB")
            protein (bool): In protein group
            auto (bool): Calculate contrast limits from data if not in metadata
        """
        dirname = channel_name
        if protein:
            dirname = f"protein/{dirname}"
        if dirname not in self.grp:
            dirname = f"{dirname}.zarr"
        assert dirname in self.grp, f"{'protein' if protein else 'channel'} {channel_name} not found"

        window = {}
        color = DEFAULT_COLORMAPS[channel_name] if channel_name in DEFAULT_COLORMAPS else DEFAULT_COLORMAPS[None]
        metadata = self.grp[f"{dirname}"].attrs

        if "omero" in metadata:
            window = metadata["omero"]["channels"][0]["window"]
            if "color" in metadata["omero"]["channels"][0]:
                color = metadata["omero"]["channels"][0]["color"]
        elif auto: # calculate limits on the fly if no ome metadata
            print(f"Could not find omero metadata for {channel_name}, calculating contrast limits now.")
            datasets = metadata["multiscales"][0]["datasets"]
            component = datasets[-1]["path"] # use top-level of pyramid for speed
            image = da.from_zarr(os.path.join(self.folder, "images", f"{dirname}"), component=component)
            window = {}
            window['min'], window['max'] = int(da.min(image)), int(da.max(image))
            window['start'],window['end'] = [int(x) for x in da.percentile(image.ravel()[image.ravel()!=0], (0.1, 99.9))]
        else:
            window['min'], window['max'] = 0, 2**16 - 1
            window['start'],window['end'] = 0, 2**16 - 1

        if window['start'] - window['end'] == 0:
            if window['end'] == 0:
                print(f"WARNING: {channel_name} image is empty.")
                window['end'] = 1000
            else:
                window['start'] = 0
        return {
            'window': window,
            'color': color
        }

    @qdebounced(timeout=400)
    def _update_omero_metadata(self, layer):
        self.grp[layer.metadata['path']].attrs['omero'] = layer.metadata['omero']
    
    def _layer_removed(self, event):
        layer = event.value
        print(f"{layer.name} removed")

    def _on_contrast_limits_changed(self, event):
        layer = event.source
        if 'omero' in layer.metadata:
            # update contrast limits
            window = {}
            window['min'], window['max'] = tuple(int(i) for i in layer.contrast_limits_range)
            window['start'], window['end'] = tuple(int(i) for i in layer.contrast_limits)
            layer.metadata['omero']['channels'][0]['window'] = window
            self._update_omero_metadata(layer)

    def _on_colormap_changed(self, event):
        layer = event.source
        if 'omero' in layer.metadata:
            # update colormap
            layer.metadata['omero']['channels'][0]['color'] = layer.colormap.name
            self._update_omero_metadata(layer)
    
    def add_channel(self, name, colormap=None):
        """Add the requested morphology channel layer

        Args:
            name (str): Name of channel
        """
        assert name in self.channels, f"{name} not one of available channels: {self.channels}"
        dirname = name if name in self.grp.group_keys() else f"{name}.zarr" 
        metadata = self.grp[f"{dirname}"].attrs
        # track updates to contrast limits and colormap
        sync_metadata = 'omero' in metadata
        try:
            metadata['path'] = dirname
        except PermissionError:
            print("Error updating metadata, local changes will not sync")
            sync_metadata = False
        datasets = metadata["multiscales"][0]["datasets"]
        im = [da.from_zarr(os.path.join(self.folder, "images", f"{dirname}"), component=d["path"]) for d in datasets]
        omero = self.omero(name)
        window = omero['window']
        if colormap is None:
            # use default for channel
            colormap = omero['color']
        elif 'omero' in metadata:
            # set as new default for channel
            metadata['omero']['channels'][0]['color'] = colormap
        layer = self.viewer.add_image(im, colormap=colormap, blending="additive", name=name,
            contrast_limits = (window['start'],window['end']),
            scale = (self.mm_per_px, self.mm_per_px),
            translate=self._top_left_mm(),
            rotate=self.rotate,
            rgb=False,
            metadata=metadata)
        layer.contrast_limits_range = window['min'],window['max']
        if sync_metadata:
            self._update_omero_metadata(layer)
            layer.events.contrast_limits.connect(self._on_contrast_limits_changed)
            layer.events.colormap.connect(self._on_colormap_changed)
        self.order_layers()

    def add_composite(self):
        """Add RGB composite layer
        """
        assert self.grp, "No zarr images directory found"
        assert 'composite' in self.grp.group_keys(), "No composite layer found, use stitch-composite to create it"
        metadata = self.grp["composite"].attrs
        datasets = metadata["multiscales"][0]["datasets"]
        im = [da.from_zarr(os.path.join(self.folder, "images", "composite"), component=d["path"]) for d in datasets]
        layer = self.viewer.add_image(im, name="Composite",
            scale = (self.mm_per_px, self.mm_per_px),
            translate=self._top_left_mm(),
            rotate=self.rotate,
            rgb=True)

    def add_protein(self, name, colormap=None, visible=True):
        """Add the requested protein expression image

        Args:
            name (str): Name of protein
        """        
        assert self.proteins, "No proteins found"
        assert name in self.proteins, f"{name} not found in proteins"
        dirname = name if name in self.grp[f"protein"].group_keys() else f"{name}.zarr"
        metadata = self.grp[f"protein/{dirname}"].attrs
        # track updates to contrast limits and colormap
        sync_metadata = 'omero' in metadata
        try:
            metadata['path'] = f"protein/{dirname}"
        except PermissionError:
            print("Error updating metadata, local changes will not sync")
            sync_metadata = False
        datasets = metadata["multiscales"][0]["datasets"]
        im = [da.from_zarr(os.path.join(self.folder, "images", "protein", dirname), component=d["path"]) for d in datasets]
        omero = self.omero(name, protein=True)
        window = omero['window']
        if colormap is None:
            # use default for channel
            colormap = omero['color']
        elif 'omero' in metadata:
            # set as new default for channel
            metadata['omero']['channels'][0]['color'] = colormap
        layer = self.viewer.add_image(im, name=name, multiscale=True,
            colormap=colormap, blending="additive",
            contrast_limits = (window['start'],window['end']),
            scale = (self.mm_per_px*(1/self.expr_scale), self.mm_per_px*(1/self.expr_scale)),
            translate=self._top_left_mm(), visible=visible,
            rotate=self.rotate,
            rgb=False,
            metadata=metadata)
        layer.contrast_limits_range = window['min'],window['max']
        if sync_metadata:
            self._update_omero_metadata(layer)
            layer.events.contrast_limits.connect(self._on_contrast_limits_changed)
            layer.events.colormap.connect(self._on_colormap_changed)
        self.order_layers()

    def add_segmentation(self):
        """Add the cell segmentation image layer
        """
        assert any("labels" in x for x in self.grp.group_keys()),\
            f"labels not found in zarr keys: {self.grp.group_keys()}"
        labelsdir = 'labels' if 'labels' in self.grp.group_keys() else 'labels.zarr'
        datasets = self.grp[labelsdir].attrs["multiscales"][0]["datasets"]
        labels = [da.from_zarr(os.path.join(self.folder, "images", labelsdir), component=d["path"]).map_blocks(
            # show edges
            find_boundaries,
        ) for d in datasets]
        cm = Colormap(['transparent', 'cyan'], controls = [0.0, 1.0])
        layer = self.viewer.add_image(labels, contrast_limits=(0, 1), colormap=cm,
            scale=(self.mm_per_px, self.mm_per_px), translate=self._top_left_mm(), blending="translucent",
            rotate=self.rotate,
            rgb=False)
        layer.opacity = 0.75
        self.segmentation_layer = layer
        layer.name = 'Segmentation'
        
    def add_cell_labels(self):
        """Add the cell labels layer
        """
        assert any("labels" in x for x in self.grp.group_keys()),\
            f"labels not found in zarr keys: {self.grp.group_keys()}"
        labelsdir = 'labels' if 'labels' in self.grp.group_keys() else 'labels.zarr'
        datasets = self.grp[labelsdir].attrs["multiscales"][0]["datasets"]
        labels = [da.from_zarr(os.path.join(self.folder, "images", labelsdir), component=d["path"]) for d in datasets]
        # TODO: need to scale for protein expression images
        layer = self.viewer.add_labels(labels, scale=(self.mm_per_px, self.mm_per_px), translate=self._top_left_mm(),
            rotate=self.rotate)
        self.cells_layer = layer
        self.cells_layer.opacity = 0.0
        self.cells_layer.visible = False
        if self.metadata is not None:
            df = self.metadata.copy()
            df['index'] = self.metadata.index
            layer.features = pd.concat([pd.DataFrame.from_dict({'index': [0]}), df])
        layer.name = 'Metadata'
        layer.metadata['label_info'] = {
            'contour': 0,
            'label_color_index': {None: 0}
        }
        layer.editable = False
  
    def cells_in_shape(self, shape_layer, idx=None):
        """Find unique cell IDs from closed shape(s)

        Cells returned are from segmentation, some may be missing in metadata.

        Args:
            shape_layer (Shapes layer): Shapes layer drawn in napari
            idx (list or int): Which shapes from layer to use, if None use all

        Returns:
            array: cell UIDs
        """
        cells = set()
        assert isinstance(shape_layer, napari.layers.Shapes), "Not a shapes layer"
        if idx is None:
            idx = range(len(shape_layer.data))
        elif not isinstance(idx, list):
            idx = [idx]
        if self.cells_layer is not None:
            for i in idx:
                if shape_layer.shape_type[i] not in ['ellipse', 'polygon', 'rectangle']:
                    continue
                mask_shape = (shape_layer.data[i] * self.mm_per_px - self._top_left_mm()) * (1/self.mm_per_px)
                dims = np.array(self.cells_layer.data.shapes)
                mask_shape = mask_shape / (dims[0]/dims[1])
                labels_arr = self.cells_layer.data[1]
                mask_shape = mask_shape.astype(int)
                ymin = max(np.min(mask_shape[:, 0]), 0)
                ymax = min(np.max(mask_shape[:, 0]), labels_arr.shape[0] - 1) 
                xmin = max(np.min(mask_shape[:, 1]), 0)
                xmax = min(np.max(mask_shape[:, 1]), labels_arr.shape[1] - 1)
                crop = labels_arr[ymin:ymax, xmin:xmax]
                mask_shape[:, 0] -= ymin
                mask_shape[:, 1] -= xmin
                if shape_layer.shape_type[i] == 'ellipse':
                    r_radius = (mask_shape[-1, 0] - mask_shape[0, 0])/2
                    r = mask_shape[0, 0] + r_radius
                    c_radius = (mask_shape[1, 1] - mask_shape[0, 1])/2
                    c = mask_shape[0, 1] + c_radius
                    i, j = ellipse(r, c, r_radius, c_radius, crop.shape)
                else:
                    r, c = mask_shape.T
                    i, j = polygon(r, c, crop.shape)
                uids = np.unique(crop.compute()[i, j])
                cells.update(uids[uids != 0])
        return np.array([int(i) for i in cells], dtype='uint32')

    def cell_target_count(self, cells):
        """Return frequency table for targets in cells

        For the given cells create frequency table
        for the targets assigned to those cells.

        Args:
            array: cell UIDs

        Returns:
            pandas.Series: series with counts and index of target
        """
        cells_df = pd.DataFrame([unpair(i) for i in cells],
                                columns=['fov', 'CellId'])
        self.targets['join_id'] = self.targets['fov'].astype(str) + '_' + self.targets['CellId'].astype(str)
        cells_df['join_id'] = cells_df['fov'].astype(str) + '_' + cells_df['CellId'].astype(str)
        df = self.targets.join(vaex.from_pandas(cells_df), on='join_id', how='inner', lsuffix='left')
        del self.targets['join_id']
        return df['target'].to_pandas_series().value_counts()

    def cells_at_points(self, points_layer):
        """Find unique cell IDs from points

        Cells returned are from segmentation, some may be missing in metadata.

        Args:
            shape_layer (Points layer): Points layer drawn in napari

        Returns:
            array: cell UIDs
        """
        assert isinstance(points_layer, napari.layers.Points), "Not a points layer"
        assert self.cells_layer is not None, "No labels layers"
        layer_coords = points_layer.data
        layer_image_coords = (layer_coords*self.mm_per_px - self._top_left_mm()) * (1/self.mm_per_px)
        layer_image_coords = layer_image_coords.astype('int')
        cells = self.cells_layer.data[0].vindex[layer_image_coords[:, 0], layer_image_coords[:, 1]].compute()
        return cells[cells != 0]

    def get_offsets(self, fov):
        """Get offsets for given FOV

        Args:
            fov (int): FOV number

        Returns:
            tuple: x and y offsets in mm
        """
        offset = self.fov_offsets[self.fov_offsets['FOV'] == fov]
        if self.dash:
            x_offset = -offset.iloc[0, ]["X_mm"]
            y_offset = offset.iloc[0, ]["Y_mm"]
        else:
            x_offset = offset.iloc[0, ]["Y_mm"]
            y_offset = -offset.iloc[0, ]["X_mm"]
        return (x_offset, y_offset)

    @property
    def channels(self):
        """List: Available morphology channels to display."""
        return self._channels

    @property
    def proteins(self):
        """List: Available protein channels to display."""
        return self._proteins

    @property
    def rotate(self):
        """float: Degrees to rotate all layers CCW for display."""
        return self._rotate

    @rotate.setter
    def rotate(self, angle):
        for i in self.viewer.layers:
            i.rotate = angle
        self._rotate = angle


    @property
    def genes(self):
        """List: If transcripts are loaded, return gene names."""
        if self.targets:
            return sorted([i for i in self.targets.category_labels('target')
                if not i.startswith('FalseCode') and
                not i.startswith('NegPrb') and
                not i.startswith('SystemControl')])
        return []

    def update_console(self):
        if self.viewer.window._qt_viewer.dockConsole.isVisible():
            self.viewer.update_console({'gem': self})
        else:
            self.viewer.window._qt_viewer.dockConsole.visibilityChanged.connect(
                lambda: self.viewer.update_console({'gem': self})
            )

    def rect_for_fov(self, fov):
        fov_height = self.fov_height*self.mm_per_px
        fov_width = self.fov_width*self.mm_per_px
        rect = np.array([
            list(self.get_offsets(fov)),
            list(map(sum, zip(self.get_offsets(fov), (fov_height, 0)))),
            list(map(sum, zip(self.get_offsets(fov), (fov_height, fov_width)))),
            list(map(sum, zip(self.get_offsets(fov), (0, fov_width))))
        ])
        y_offset = self._top_left_mm()[0]
        x_offset = self._top_left_mm()[1]
        return [[i[0] - y_offset, i[1] - x_offset] for i in rect]

    # keep reference to layer and keep on top
    # keep selected FOVs in object
    def add_fov_labels(self):
        rects = [self.rect_for_fov(i) for i in self.fov_offsets['FOV']]
        shape_properties = {
            'label': self.fov_offsets['FOV'].to_numpy()
        }
        text_parameters = {
            'text': 'label',
            'size': 12,
            'color': 'white'
        }
        shapes_layer = self.viewer.add_shapes(rects,
            face_color='#90ee90',
            edge_color='white',
            edge_width=0.02,
            properties=shape_properties,
            text = text_parameters,
            name = 'FOV labels',
            translate=self._top_left_mm(),
            rotate=self.rotate)
        shapes_layer.opacity = 0.5
        shapes_layer.editable = False
        self.fov_labels = shapes_layer

    def center_fov(self, fov:int, buffer:float=1.0):
        """Center FOV in canvas and zoom to fill

        Args:
            fov (int): FOV number
            buffer (float): Buffer size for zoom. < 1 equals zoom out.
        """        
        extent = [np.min(self.rect_for_fov(fov), axis=0) + self._top_left_mm(),
            np.max(self.rect_for_fov(fov), axis=0) + self._top_left_mm()]
        size = extent[1] - extent[0]
        self.viewer.camera.center = np.add(extent[0], np.divide(size, 2))
        self.viewer.camera.zoom = np.min(np.array(self.viewer._canvas_size) / size) * buffer
        
    def move_to_top(self, layer):
        """Move layer to top of layer list. Layer must be in list.

        Args:
            layer (Layer): Layer to move
        """        
        idx = self.viewer.layers.index(layer)
        self.viewer.layers.move(idx, -1)

    def order_layers(self):
        if self.cells_layer is not None:
            self.move_to_top(self.cells_layer)
        if self.color_cells_layer is not None:
            self.move_to_top(self.color_cells_layer)
        points_layers = [l for l in self.viewer.layers if isinstance(l, napari.layers.Points)]
        for l in points_layers:
            self.move_to_top(l) 
        if self.segmentation_layer is not None:
            self.move_to_top(self.segmentation_layer)
        if self.fov_labels is not None:
            self.move_to_top(self.fov_labels)
        if self.cells_layer is not None:
            self.viewer.layers.selection.active = self.cells_layer

    def read_metadata(self, path):
        df = pd.read_csv(path)
        if not 'UID' in df:
            assert 'cell_ID' in df.columns, "Need UID or cell_ID column in metadata"
            res = [re.search("c_.*_(.*)_(.*)", i) for i in df['cell_ID']]
            df['UID'] = [pair(int(i.group(1)), int(i.group(2))) for i in res]
        df.set_index('UID', inplace=True)   
        self.metadata = df
        if self.cells_layer is not None:
            # update features
            df = self.metadata.copy()
            df['index'] = self.metadata.index
            self.cells_layer.features = pd.concat([pd.DataFrame.from_dict({'index': [0]}), df])
    
    def export_omero(self, path=None):
        """Export image metadata including colormap and contrast limits for all available channels.

        Args:
            path (str, optional): If provided write results to CSV. Defaults to None.

        Returns:
            DataFrame: Image metadata in pandas DataFrame.
        """
        omero = [self.omero(i, auto=False) for i in self.channels]
        df = pd.DataFrame(data={
            'name': self.channels,
            'type': 'channel',
            'color': [i['color'] for i in omero],
            'start': [i['window']['start'] for i in omero],
            'end': [i['window']['end'] for i in omero]
        })
        if self.proteins:
            omero = [self.omero(i, protein=True, auto=False) for i in self.proteins]
            protein_df = pd.DataFrame(data={
                'name': self.proteins,
                'type': 'protein',
                'color': [i['color'] for i in omero],
                'start': [i['window']['start'] for i in omero],
                'end': [i['window']['end'] for i in omero]
            })
            df = pd.concat([df, protein_df])
        if path:
            df.to_csv(path, index=False)
        return df
    
    def is_categorical_metadata(self, col_name):
        return not is_numeric_dtype(self.metadata[col_name]) or len(pd.unique(self.metadata[col_name])) < 30

    def color_cells(self, col_name, color=None, contour=0, subset=None):
        """Change cell labels layer based on metadata

        Args:
            col_name (str): Column name in metadata, "all" colors cells the same color.
            color (str|dict): (1) Color name if col_name is "all", or
                (2) dictionary with keys being metadata values and value being color name, or
                (3) colormap for continuous metadata, https://matplotlib.org/stable/tutorials/colors/colormaps.html
            contour (int): Labels layer contour, 0 is filled, otherwise thickness of lines.
            subset (list of int): List of cell UIDs, if given only color these cells.
        """
        label_colors = {}
        if self.metadata is None:        
            assert col_name == "all", "No metadata loaded"
            label_colors = {1: color, None: color}
        else:
            cells = subset if (subset is not None) else self.metadata.index
            if col_name == "all":
                color = transform_color(color)[0]
                label_colors = {k:color for k in cells}
            else:
                assert col_name in self.metadata, f"{col_name} not in metadata"
                if self.is_categorical_metadata(col_name):
                    if color is None:
                        if self.adata is not None and col_name + "_colors" in self.adata.uns:
                            # get colors from AnnData object
                            categories = self.adata.obs[col_name].cat.categories
                            cat_colors = [transform_color(i)[0] for i in self.adata.uns[col_name + "_colors"]]
                            color = dict(zip(categories, cat_colors))
                        else:
                            vals = np.unique(self.metadata[col_name])
                            cm = label_colormap(len(vals)+1)
                            color = dict(zip(vals, cm.colors[1:]))
                    else:
                        assert isinstance(color, dict), "color needs to be dict for categorical metadata"
                        color = {k:transform_color(v)[0] for k,v in color.items()}
                    label_colors = {k:(color[v] if v in color else transform_color('transparent')[0])
                        for k,v in zip(cells, self.metadata.loc[cells][col_name])}
                else:
                    if color is None:
                        color = 'gray'
                    assert color in AVAILABLE_COLORMAPS, f"{color} not in {AVAILABLE_COLORMAPS.keys()}"
                    cm = AVAILABLE_COLORMAPS[color]
                    min_max_scaler = preprocessing.MinMaxScaler()
                    # normalize to 0-1 with full range present in metadata
                    x = pd.Series(
                        min_max_scaler.fit_transform(self.metadata[col_name].values.reshape(-1, 1))[:, 0],
                        self.metadata.index,
                        dtype=np.float32
                    )
                    label_colors = {k:v for k,v in zip(cells, cm.map(x.loc[cells]))}
            label_colors[None] = transform_color('transparent')[0]
        name = 'Cells' if col_name == 'all' else col_name
        # when labels are fast in napari can set labels layer colors directly
        self._color_cells(colors=label_colors, name=name, contour=contour)
        self.order_layers()

    def _map_labels_to_colors(self, arr):
        contour = self.cells_layer.metadata['label_info']['contour']
        label_color_index = self.cells_layer.metadata['label_info']['label_color_index']
        # following logic in napari Labels layer
        arr_modified = arr
        if contour > 0:
            arr_modified = np.zeros_like(arr)
            struct_elem = ndi.generate_binary_structure(arr.ndim, 1)
            thick_struct_elem = ndi.iterate_structure(
                struct_elem, contour
            ).astype(bool)
            boundaries = ndi.grey_dilation(
                arr, footprint=struct_elem
            ) != ndi.grey_erosion(arr, footprint=thick_struct_elem)
            arr_modified[boundaries] = arr[boundaries]
        u, inv = np.unique(arr_modified, return_inverse=True)
        image = np.array(
            [
                label_color_index[x]
                if x in label_color_index
                else label_color_index[None]
                for x in u
            ]
        )[inv].reshape(arr_modified.shape)
        return image

    def _color_cells(self, colors, name='Cells', contour=0):
        if self.color_cells_layer in self.viewer.layers:
            self.viewer.layers.remove(self.color_cells_layer)
        assert any("labels" in x for x in self.grp.group_keys()),\
            f"labels not found in zarr keys: {self.grp.group_keys()}"
        labelsdir = 'labels' if 'labels' in self.grp.group_keys() else 'labels.zarr'
        datasets = self.grp[labelsdir].attrs["multiscales"][0]["datasets"]
        # from napari Labels layer
        with warnings.catch_warnings():
            # suppress known warning
            warnings.filterwarnings(action='ignore', message='.*distinct colors.*')
            custom_colormap, label_color_index = color_dict_to_colormap(colors)
        self.cells_layer.metadata['label_info']['contour'] = contour
        self.cells_layer.metadata['label_info']['label_color_index'] = label_color_index
        im = [da.from_zarr(os.path.join(self.folder, "images", labelsdir), component=d["path"]).map_blocks(
            self._map_labels_to_colors,
            dtype=float
        ) for d in datasets]
        layer = self.viewer.add_image(im,
            scale=(self.mm_per_px, self.mm_per_px), translate=self._top_left_mm(), blending="translucent",
            rotate=self.rotate,
            contrast_limits=(0,1),
            interpolation='nearest',
            colormap=custom_colormap,
            cache=True,
            rgb=False)
        layer.name = name
        layer.opacity = 1.0
        self.color_cells_layer = layer

    def plot_transcripts(self, gene, color, point_size=5):
        """Plot targets as dots

        Args:
            gene (str): Target to plot
            color (str): Color for points.
            point_size (int, optional): Point size. Defaults to 5.

        Returns:
            napari.layers.Points: A points layer for transcripts.
        """
        assert self.targets, "No targets found, use read_targets.py first to create targets.hdf5 file."
        self.targets.select(self.targets.target == gene)
        y = self.targets.evaluate(self.targets.global_y, selection=True)
        x = self.targets.evaluate(self.targets.global_x, selection=True)
        points = np.array(list(zip(y, x)))
        points_layer = self.viewer.add_points(points,
            size = point_size,
            edge_width=0,
            face_color=color,
            scale = (self.mm_per_px, self.mm_per_px),
            translate=self._top_left_mm(),
            rotate=self.rotate)
        points_layer.name = gene
        points_layer.opacity = 1.0
        self.order_layers()
        return points_layer

    def add_points(self, fov=None, color=None, gray_no_cell=None, point_size=1):
        """Add all targets

        Args:
            fov (int or list, optional): Only add points for specified fov(s).
            color (str, optional): Color for points. If None will color by gene.
            gray_no_cell (bool, optional): Whether to color targets with no CellID gray. Defaults to True if color specified.
            point_size (int, optional): Point size. Defaults to 1.

        Returns:
            napari.layers.Points: A points layer for transcripts.
        """
        assert self.targets, "No targets found, use read_targets.py first to create targets.hdf5 file."
        if gray_no_cell is None:
            gray_no_cell = color is not None
        targets = self.targets[~self.targets.target.str.startswith("NegPrb") &
            ~self.targets.target.str.startswith("FalseCode") &
            ~self.targets.target.str.startswith("SystemControl")]
        if fov is not None:
            targets = targets[targets.fov.isin(fov if isinstance(fov, list) else [fov])]
        y = targets.global_y.evaluate()
        x = targets.global_x.evaluate()
        id = targets.CellId.evaluate()
        genes = targets.target.evaluate()
        u, inv = np.unique(genes, return_inverse=True)
        cm = label_colormap(len(self.targets.category_labels('target'))+1)
        face_color = cm.colors[1:][inv] if color is None else np.full(shape=(len(genes), 4), fill_value=transform_color(color))
        if gray_no_cell:
            face_color[id == 0] = transform_color('gray')
        points = np.array(list(zip(y,x)))
        print(f"Plotting {len(genes):,} transcripts")
        points_layer = self.viewer.add_points(points,
            size = point_size,
            edge_width=0,
            properties={
                'CellId': id,
                'Gene': genes
            },
            scale=(self.mm_per_px, self.mm_per_px),
            face_color=face_color,
            translate=self._top_left_mm(),
            rotate=self.rotate)
        points_layer.opacity = 0.7
        points_layer.name = "Targets"
        return points_layer

    def layers_to_metadata(self:'napari_cosmx.gemini.Gemini', layer_names:'list')-> 'pandas.core.frame.DataFrame':
        """Create metadata columns to indicate if cells overlap with the shapes or points in layers.

        Args:
            layer_names (list): Names of shapes or points layers, layer names will be used as column names in metadata.

        Returns:
            pandas.core.frame.DataFrame: Metadata object with additional columns appended, True if cell overlaps with shape, False otherwise.
        """
        assert self.metadata is not None, "No metadata found"
        meta = self.metadata.copy()
        invalid = [x for x in layer_names if x not in self.viewer.layers]
        assert not invalid, f"No layers found named {invalid}."
        layers = [self.viewer.layers[x] for x in layer_names]
        invalid = [x for x in layers if not isinstance(x, napari.layers.Shapes) and not isinstance(x, napari.layers.Points)]
        assert not invalid, f"These layers are not Shapes or Points layers: {invalid}."
        for layer in layer_names:
            meta[layer] = False
            if isinstance(self.viewer.layers[layer], napari.layers.Shapes):
                cells_in_layer = self.cells_in_shape(self.viewer.layers[layer])
            elif isinstance(self.viewer.layers[layer], napari.layers.Points):
                cells_in_layer = self.cells_at_points(self.viewer.layers[layer])
            meta.loc[meta.index.isin(cells_in_layer), layer] = True
        return meta

    def save_layers(self:'napari_cosmx.gemini.Gemini', layer_names:'list', out_dir:'str'=None):
        """Save Shapes and Points layers as pickle objects. Re-open with load_layer.

        Args:
            layer_names (list): The names of the layers to save.
            out_dir (str, optional): Directory to save pickle file. Defaults to slide folder.
        """
        if out_dir is None:
            out_dir = self.folder
        invalid = [x for x in layer_names if x not in self.viewer.layers]
        assert not invalid, f"No layers found named {invalid}."
        layers = [self.viewer.layers[x] for x in layer_names]
        invalid = [x for x in layers if not isinstance(x, napari.layers.Shapes) and not isinstance(x, napari.layers.Points)]
        assert not invalid, f"These layers are not Shapes or Points layers: {invalid}."
        for layer_name in layer_names:
            print('Saving ' + layer_name + ' to ' + out_dir)
            layer_tuple = self.viewer.layers[layer_name].as_layer_data_tuple()
            # save a pickle
            with open(os.path.join(out_dir, layer_name+'.pickle'), 'wb') as f:
                pickle.dump(layer_tuple, f)

    def load_layers(self:'napari_cosmx.gemini.Gemini', layer_names:'list'=[], input_dir:'str'=None):
        """Loads previously saved layer (pickle file) into napari.

        Args:
            layer_names (list, optional): If not empty, load only these layers as long as they end with ".pickle". Defaults to [].
            input_dir (str, optional): directory of the pickle files. Defaults to slide folder.
        """
        if input_dir is None:
            input_dir = self.folder
        if len(layer_names) == 0:
            layer_names = [i for i in listdir(input_dir) if i.endswith(".pickle")]
        else:
            layer_names = [i+'.pickle' for i in layer_names]
        
        for layer_name in layer_names:
            with open(os.path.join(input_dir, layer_name), 'rb') as f:
                layer_tuple = pickle.load(f)
            layer_metadata = layer_tuple[1]
            print('Adding ' + layer_tuple[2] + ' for ' + layer_name)

            if layer_tuple[2] == 'shapes':
                valid_kwargs = signature(self.viewer.add_shapes).parameters
            elif layer_tuple[2] == 'points':
                valid_kwargs = signature(self.viewer.add_points).parameters

            layer_kwargs = {k:v for k,v in layer_metadata.items() if k in valid_kwargs}
            if version('napari') <= '0.4.16':
                layer_kwargs['text']['color'] = None

            if layer_tuple[2] == 'shapes':
                self.viewer.add_shapes(data=layer_tuple[0], **layer_kwargs)
            elif layer_tuple[2] == 'points':
                self.viewer.add_points(data=layer_tuple[0], **layer_kwargs)

