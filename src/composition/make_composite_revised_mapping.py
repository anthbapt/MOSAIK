# Â©2024 Bruker Spatial Biology, Inc. All rights reserved. Subject to additional license terms and conditions provided separately by Bruker Spatial Biology, Inc.

#!/usr/bin/env python
# coding: utf-8
"""
Application: make_composite.py
Author: Vikram Kohli, PSS
Version: 1.2.2

Description:
The script creates composite images from the layered morphology 2D images.
Layered images are extracted from the 2D morphology tif files and written in a file format selected by the user.
The extracted images are converted to 8bit, and composite images are written from these 8bit images.

User inputs: 
clipping - Histogram clipping percentage. This value is the percentage of the histogram to clip on the left
and right side. The effect changes the contrast of the image. Higher % produces more contrast.

user_format - File format to be written. Options are jpg, png, and tif.

Output:
raw_folder - Extracted images from the layered 2D morphology images
bit_reduced_folder - The images in the raw_folder are converted to 8bit
bit_reduced_autocontrast - Imanges in the bit_reduced_folder are autocontrasted based on clipping parameter.
composite_folder - Composite image from the files in the bit_reduced_folder. The composite is a screen composite.
composite_autocontrast_folder - Composite images autocontrasted based on the clipping parameter.


"""
#Loading required libraries
from PIL import Image, ImageSequence, ImageOps, ImageChops
import numpy as np
import glob
import os
import sys
import re
import shutil
from os import name
from time import perf_counter

tiff_dir = '/Users/k2481276/Documents/CosMx/G16_sopa/Morphology2D/'
os.chdir(tiff_dir)

#Allowed image formats
image_formats = ['jpg', 'png', 'tif']

#The list of ouptut folders
raw_folder = 'raw'
bit_reduced_folder = '8bit'
bit_reduced_autocontrast_folder = '8bit_autocontrast'
composite_autocontrast_folder = 'composite_autocontrast'
composite_folder = 'composite'

#Composite color scheme. Color are listed in order of channel number, from channel 0 to channel 4.
colors = ['magenta', 'yellow','cyan', 'green', 'blue']
#colors = ['green', 'yellow', 'gray', 'red', 'blue']

marker_names = ['Marker1', 'Marker2', 'Marker3', 'Marker4', 'Marker5']  # 替换为您的实际marker名称
marker_channel_map = {i: marker for i, marker in enumerate(marker_names)}
marker_color_map = {marker: color for marker, color in zip(marker_names, colors)}

#Morphology 2D images have a particular naming format. A regex pattern match is performed to select only those
pattern = r"([\d]+_[\d]+_[\w]+_[\w]+_[\w]+_[\w]+_[F]+[\d]*)"
regex = re.compile(pattern, flags = re.IGNORECASE)

#Lossless file compression value. Higher values produce smaller files at the expensive of increased script
#execution time. The set value is a compromise between file size and execution time. 
compress_value = 3

# wrong input
class ArgumentException(Exception): 
    
    '''Exception class for handling argument errors. The exception is raised when invalid argument types
       are found. '''
# This saves the error message and tells Python it's a real error.       
    def __init__(self, message):
        self.message = message
        super().__init__(message)
# folder already exists
class FolderException(Exception):
    
    '''Exception class for handling folder exceptions. The exception is raised when folders are already present.'''
    
    def __init__(self, message):
        self.message = message
        super().__init__(message)
# ask the user to enter clipping value and user format     
def check_arguments(image_formats):
    
    '''Checks user input values. ValueError is raised if clipping_input is not of int or float type.
       Argument excpetion raised if the user_format is not of jpg, png, or tif type. The script terminates
       on ValueError or raised argument exception'''
       
    clipping_input = input('\nPlease specify a clipping percentage as an integer or float: ').strip()
# stop the programme when the user is not entering in the right way    
    try:
        clipping_input = float(clipping_input)
    except ValueError:
        print('Value Error: clipping value must be of type integer or float and not null...Exiting')
        sys.exit(1)
    
    user_format = input('\nPlease specify one of the allowed image format types [jpg, png, tif]:  ' ).lower().strip()
       
    if user_format not in image_formats:
        message = '\n\nImage format type error - Allowed formats include: jpg, png, tif...Exiting'
        raise ArgumentException(message)
        sys.exit(1)    
# Give back the two values the user typed.   
    return (clipping_input, user_format)
# check if the folders already exist
def FoldersCheckAndCreate():
    
    ''' Function creates the output folders. If the folders already exist an exception is raised. 
        OSError is raised if the folders cannot be created. The script terminates on raised exceptions.'''
    
    if os.path.exists(raw_folder):
        message = '\nRaw folder already exists...Exiting'
        raise FolderException(message)
        sys.exit(1)
    elif os.path.exists(bit_reduced_folder):
        message = '\n8bit folder already exits...Exiting'
        raise FolderException(message)
        sys.exit(1)
    elif os.path.exists(bit_reduced_autocontrast_folder):
        message = '\n8bit autocontrast folder already exists...Exiting'
        raise FolderException(message)
        sys.exit(1)
    elif os.path.exists(composite_autocontrast_folder):
        message = '\nComposite autocontrast folder already exitst...Exiting'
        raise FolderException(message)
        sys.exit(1)
    elif os.path.exists(composite_folder):
        message = '\nComposite folder already exitst'
        raise FolderException(message)
        sys.exit(1)
    else:
        folder_list = ['raw', '8bit', '8bit_autocontrast', 'composite_autocontrast', 'composite']
        try:
            for folder in folder_list:
                os.mkdir(folder)
            print('Folders successfully created!')
        except OSError:
            print('Issues creating one or more folders...Exiting')
            sys.exit(1)
                      
def get_global_channel_max_values(tiff_files, regex):
    channel_max_values = {} 
    
    for image_file in tiff_files:
        if regex.match(image_file):
            image = Image.open(image_file)
            for channel, layer in enumerate(ImageSequence.Iterator(image)):
                array = np.array(layer)
                current_max = array.max()
                
                if channel not in channel_max_values or current_max > channel_max_values[channel]:
                    channel_max_values[channel] = current_max
    
    for channel in range(5): 
        if channel not in channel_max_values or channel_max_values[channel] == 0:
            channel_max_values[channel] = 1.0
            
    return channel_max_values

# Turn the image pixel values into 8-bit grayscale
def force_8bit(image_file, channel=None, global_max=None):
    
    '''The function converts the image file to 8bit. If the user_format is jpg, this function is skipped.
       Numpy is used for fast vectorization. '''
    # Turn image into numbers we can calculate with(np) / Convert the image to a NumPy array for easier numerical processing. Normalize the pixel values to 0-255.    
    array = np.array(image_file)
    
    if global_max is not None and channel is not None:
        if global_max > 0:
            reduced_bit = (array / global_max) * 255
        else:
            reduced_bit = (array / 1.0) * 255
    else:
        if array.max() == 0:
            reduced_bit = (array / 1.0) * 255
        else:
            reduced_bit = (array / array.max()) * 255
            
    # Turn the numbers back into an image.        
    img = Image.fromarray(reduced_bit.astype('uint8'))
    return img

# extract each layer from a .TIF file and saves it separately        
def layer_extraction(image, fov_num, current_dir, raw_folder, user_format):
    
    '''Function extracts the layers from the 2D morphology tif files and writes the files to the raw folder.
       If the user_fomrat is jpg, the extracted layers are converted to 8bit before saving.'''
    # Go into the folder where raw images should be saved    
    os.chdir(raw_folder)
    # Scale the pixel values of each image layer by 1/256 (converting from 16-bit to 8-bit), then save it as a grayscale image.  
    if user_format == 'jpg':       
        for channel, layer in enumerate(ImageSequence.Iterator(image)):
            marker = marker_channel_map.get(channel, f"ch{channel}")
            layer.point(lambda value: value*(1./256)).convert('L').save(f'{fov_num}_{marker}_raw.{user_format}', compress_type = compress_value)
        os.chdir(current_dir)
    else:    
        for channel, layer in enumerate(ImageSequence.Iterator(image)):
            marker = marker_channel_map.get(channel, f"ch{channel}")
            layer.save(f'{fov_num}_{marker}_raw.{user_format}', compress_type = compress_value)
        os.chdir(current_dir)

# Overlay multiple channel images with color to create a single composite color image.
def write_composite(image_files, colors):   
    
    '''Fucntion creates a screen composite image with colors specified by the colors list variable.'''
# Start with the last image file, color it, and use it as the base.    for...:For each other channel-color it and blend it using "screen" mode.  
    out = image_files[-1]
    out = Image.open(out)
    img_color = colors[-1]
    out = ImageOps.colorize(out, black = 'black', white = img_color)
    for image_file, color in zip(image_files[::-1][1:], colors[::-1][1:]):
        img = Image.open(image_file)
        img = ImageOps.colorize(img, black = 'black', white = color)
        out = ImageChops.screen(out, img)
    return out



if __name__ == '__main__':
    
    if name == 'nt':
        os.system('cls') #Windows only
    else:
        os.system('clear') #Posix only
        
    current_dir = os.getcwd()
    
    print('********************************')
    print('   Composite script started     ')
    print('********************************\n')
    clipping, user_format = check_arguments(image_formats) #Request the clipping and file format from the user
    print(f'\nSetting the clipping value for autocontrast to {clipping}% and the export format to {user_format}')
    print('\nChecking for existing folders and creating the necessary folders..')
    
    FoldersCheckAndCreate()    #Check if the image folders exist. If not create the folders.
         
    start = perf_counter() #Used in calaculating the script exectution time. 
    
    
    print(f'\nPlease wait: Processing the files in {current_dir}')
    
    tiff_files = glob.glob('*.tif') + glob.glob('*.TIF')
    matching_tiff_files = [f for f in tiff_files if regex.match(f)]
    
    print(f'\n...Analyzing all images to standardize channel intensities')
    channel_max_values = get_global_channel_max_values(matching_tiff_files, regex)
    print(f'Global maximum values for each channel: {channel_max_values}')
    
    #Loop through all images that match the regex mattern and extract images from the layered 2D morphology files
    print(f'\n...Extracting {user_format} files (raw) from layered 2D morphology image files')   
    for image_file in matching_tiff_files:
        fov_num = image_file.split('_')[-1].replace('.TIF','')
        image = Image.open(image_file)
        layer_extraction(image, fov_num, current_dir, raw_folder, user_format) #Calls the function to extract the layered images from the 2D morphology tif files.
         
    #Convert images to 8bit. If the user_format is jpg, conversion is skipped and the files are copied to the 8bit folder        
    print(f'\n...Converting raw files to 8bit of type {user_format} and creating 8bit autocontrast files')  
    if user_format == 'jpg':      
        print(f'** {user_format.title()} files already in 8bit format...copying files to 8bit folder **')
        os.chdir(raw_folder)
        jpg_files = [files for files in glob.glob('*.' + user_format)]
        
        #Copy jpg files in raw_folder to the bit_reduced folder
        for jpg in jpg_files:
            shutil.copy2(os.path.join(os.getcwd(),jpg), os.path.join(current_dir, bit_reduced_folder))
        os.chdir(os.path.join(current_dir, bit_reduced_folder))
        
        #Write autocontrasted 8bit image files
        for image_file in glob.glob('*.' + user_format):
             image = Image.open(image_file)
             os.chdir(os.path.join(current_dir, bit_reduced_autocontrast_folder))
             ImageOps.autocontrast(image, cutoff = clipping, ignore = None, preserve_tone = False).save(f"{image_file.replace('_raw', '_8bit_autocontrast')}", compress_type = compress_value)
             os.chdir(os.path.join(current_dir, raw_folder))            
        os.chdir(current_dir)
    
    else:    #If user_format is not jpg, convert the images to 8bit and write autocontrast 8bit files.
        os.chdir(raw_folder)
        for image_file in glob.glob('*.' + user_format):
            img = Image.open(image_file)
            file_parts = os.path.basename(image_file).split('_')
            marker_name = file_parts[1]  
            
            channel = None
            for ch, marker in marker_channel_map.items():
                if marker == marker_name:
                    channel = ch
                    break
            
            if channel is not None and channel in channel_max_values:
                img = force_8bit(img, channel, channel_max_values[channel]).convert('L')
            else:
                img = force_8bit(img).convert('L')
                
            os.chdir(os.path.join(current_dir, bit_reduced_folder))
            img.save(f"{image_file.replace('_raw', '_8bit')}", compress_type = compress_value)          
            os.chdir(os.path.join(current_dir, bit_reduced_autocontrast_folder))
            ImageOps.autocontrast(img, cutoff = clipping, ignore = None, preserve_tone = False).save(f"{image_file.replace('_raw', '_8bit_autocontrast')}", compress_type = compress_value)           
            os.chdir(os.path.join(current_dir, raw_folder))
        os.chdir(current_dir)
    
    #Write the composite and autocontrast composite images.
    print(f'\n...Writing composite and composite autocontrast images as {user_format}')           
    os.chdir(bit_reduced_folder)
    fov_num = [fov.split('_')[0] for fov in glob.glob('*.' + user_format)]
    fov_num = list(set(fov_num))
    
    for fov in fov_num:
        composite_files = []
        composite_colors = []
        
        for marker in marker_names:
            marker_files = glob.glob(f'{fov}_{marker}_*.' + user_format)
            if marker_files:
                composite_files.append(marker_files[0])
                composite_colors.append(marker_color_map[marker])
        
        
        if composite_files:
            try:
                composite_image = write_composite(composite_files, composite_colors)
                os.chdir(os.path.join(current_dir, composite_folder))
                composite_image.save(f'{fov}_composite.' + user_format, compress_type = compress_value)
                os.chdir(os.path.join(current_dir, composite_autocontrast_folder))
                ImageOps.autocontrast(composite_image, cutoff = clipping, ignore = None, preserve_tone = False ).save(f'{fov}_composite_autocontrast.' + user_format, compress_type = compress_value)
                os.chdir(os.path.join(current_dir, bit_reduced_folder))
            except Exception as e:
                print(f"处理出错: {str(e)}")
    
    end = perf_counter() #Used in calcualting script execution time.
       
    print(f'\nFinished. Total run time = {int(end - start)} sec')

