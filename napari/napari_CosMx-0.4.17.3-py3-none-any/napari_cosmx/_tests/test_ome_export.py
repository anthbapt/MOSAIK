import pytest
from unittest.mock import MagicMock  # For the less recommended approach (if not using pytest-mock)
import numpy as np
import zarr
import dask.array as da
from tifffile import TiffWriter, imread
from pathlib import Path
import os
import re
from napari_cosmx.utils.export_tiff import split_list_element, BatchStorage, _edges, _scale_edges, main

test_data_location = os.path.join('data', 'Liver-S2')

def test_split_list_element():
    assert split_list_element(["a,b,c"]) == ["a", "b", "c"]
    assert split_list_element(["a b c"]) == ["a", "b", "c"]
    assert split_list_element(["a, b,c"]) == ["a", "b", "c"]
    assert split_list_element(["a"]) == ["a"]
    with pytest.raises(TypeError):  # Correctly test for TypeError
        split_list_element()
    assert split_list_element([""]) == []
    assert split_list_element([]) == []
    assert split_list_element("a") == "Input must be a list"
    assert split_list_element(["a", "b"]) == "Input list must have length 1"
    assert split_list_element(0) == "Input must be a list"


def test_batch_storage():
    storage = BatchStorage(2)
    storage.add_item("item1")
    storage.add_item("item2")
    storage.add_item("item3")
    assert len(storage.storage) == 2
    assert storage.get_batch(0) == ['item1', 'item2']
    assert storage.get_batch(1) == ['item3']

    assert str(print(storage)) == "None"

    storage_with_labels = BatchStorage(2)
    labels = "labels_data"
    storage_with_labels.set_labels(labels)
    storage_with_labels.add_item("item1")
    storage_with_labels.add_item("item2")
    storage_with_labels.add_item("item3")
    assert storage_with_labels.get_batch(0) == ['labels_data', 'item1', 'item2']
    assert storage_with_labels.get_batch(1) == ['labels_data', 'item3']


def test_edges():
    x = np.array([[1, 1, 2], [1, 1, 1], [1,1,1]])
    edges = _edges(x)
    expected = np.array([[0, 1, 1], [0, 1, 1], [0,0,0]], dtype=np.uint8)
    np.testing.assert_array_equal(edges, expected)

def test_scaling():
    x = np.array([[1, 1, 2], [1, 1, 1], [1,1,1]])
    edges = _edges(x)
    scaled = _scale_edges(edges)
    expected = np.array([[1e2, 1e4, 1e4], [1e2, 1e4, 1e4], [1e2,1e2,1e2]], dtype=np.uint8)    


@pytest.mark.skipif(not os.path.exists(test_data_location), reason="Test data not found. Skipping.") #Skip test if data is not found
def test_main():

    # Create a dummy argparse object (since we're not using command-line args directly)
    class MockArgs:
        def __init__(self, inputdir, outputdir, filename, compression="zlib", batchsize=2, segmentation=False, channels=None, proteins=None, levels=8, verbose=False, libvips=False, vipshome=None, vipsconcurrency=8):
            self.inputdir = inputdir
            self.outputdir = outputdir
            self.filename = filename
            self.compression = compression
            self.batchsize = batchsize
            self.segmentation = segmentation
            self.channels = channels
            self.proteins = proteins
            self.levels = levels
            self.verbose = verbose
            self.libvips = libvips
            self.vipshome = vipshome
            self.vipsconcurrency = vipsconcurrency
    
    # Must be a valid zarr store
    inputdir = "not_a_valid_zarr_path" # test_data_location
    outputdir = "test_output1"  # Create a temporary output directory if needed
    filename = "test.ome.tif"
    batchsize = 1

    args1 = MockArgs(inputdir, outputdir, filename, batchsize)  # mock arguments

    with pytest.raises(SystemExit) as excinfo:
        main(args1)

    assert "Could not find images directory at not_a_valid_zarr_path" in str(excinfo.value)

    # Should work
    args2 = MockArgs(inputdir = test_data_location, 
                     outputdir = outputdir, 
                     filename = filename,
                     segmentation = True, 
                     channels = ["foo DNA"])
    
    main(args2)
    assert os.path.exists(os.path.join(outputdir, f"batch_0_{filename}"))

    # no IFs valide
    args3 = MockArgs(inputdir = test_data_location, 
                     outputdir = "test_output3", 
                     filename = filename,
                     segmentation = True, 
                     channels = ["no valid IFs"], 
                     verbose = True)
    
    main(args3)
    assert os.path.exists(os.path.join("test_output3"))
    assert not os.path.exists(os.path.join("test_output3", f"batch_0_{filename}"))

    args4 = MockArgs(inputdir = test_data_location, 
                     outputdir = "test_output4", 
                     filename = filename,
                     segmentation = True, 
                     channels = [], 
                     verbose = True,
                     libvips = True)

    main(args4)
    assert os.path.exists(os.path.join("test_output4", f"batch_0_{filename}"))







