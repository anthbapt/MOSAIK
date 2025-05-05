import tifffile
import numpy as np
import glob
import os

path = '/Users/k2481276/Documents/CosMx/G16_io_sopa'
in_folder = path + "/Morphology2D"
out_folder = path + "/Morphology2D_out"
os.chdir(path)

list_tiff = sorted(glob.glob(in_folder + "/*.TIF"))

try:
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print(f"Folder created: {out_folder}")
    else:
        print(f"Folder already exists: {out_folder}")
except Exception as e:
    print(f"An error occurred: {e}")
        

def remove_slice_from_tiff(input_path, output_path, slice_index_to_remove):
    # Load the multi-slice TIFF
    with tifffile.TiffFile(input_path) as tif:
        images = tif.asarray()

    # Check the number of slices
    num_slices = images.shape[0]
    if slice_index_to_remove < 0 or slice_index_to_remove >= num_slices:
        raise ValueError(f"Slice index must be between 0 and {num_slices - 1}")

    # Remove the specified slice
    new_images = np.delete(images, slice_index_to_remove, axis=0)
    name = input_path.split("/")[-1]
    # Save the modified image stack
    tifffile.imwrite(output_path + "/" + name, new_images)

    print(f"Slice {slice_index_to_remove} removed. New file saved to {output_path}")

for k in list_tiff:
    remove_slice_from_tiff(k, out_folder, 4)