#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 15:06:30 2025

@author: niuyiren
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

#Morphology 2D images have a particular naming format. A regex pattern match is performed to select only those
#images that match the format. 
pattern = r"([\w]+_[\w]+_[\w]+_[\w]+_[\w]+_[\w]+_[F]+[\d]*)"
regex = re.compile(pattern, flags = re.IGNORECASE)

#Lossless file compression value. Higher values produce smaller files at the expensive of increased script
#execution time. The set value is a compromise between file size and execution time. 
compress_value = 3

# wrong input
class ArgumentException(Exception): 
    
    '''Exception class for handling argument errors. The exception is raised when invalid argument types
       are found. '''
# This saves the error message and tells Python it’s a real error.       
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
                      
# extract each layer from a .TIF file and saves it separately        
def layer_extraction(image, fov_num, current_dir, raw_folder, user_format):
    
    '''Function extracts the layers from the 2D morphology tif files and writes the files to the raw folder.
       If the user_fomrat is jpg, the extracted layers are converted to 8bit before saving.'''
# Go into the folder where raw images should be saved    
    os.chdir(raw_folder)
# Scale the pixel values of each image layer by 1/256 (converting from 16-bit to 8-bit), then save it as a grayscale image.  
    if user_format == 'jpg':       
        for channel, layer in enumerate(ImageSequence.Iterator(image)):           
            layer.point(lambda value: value*(1./256)).convert('L').save(f'{fov_num}_ch{channel}_raw.{user_format}', compress_type = compress_value)
        os.chdir(current_dir)
    else:    
        for channel, layer in enumerate(ImageSequence.Iterator(image)):
            layer.save(f'{fov_num}_ch{channel}_raw.{user_format}', compress_type = compress_value)
        os.chdir(current_dir)

# Turn the image pixel values into 8-bit grayscale
def force_8bit(image_file):
    
    '''The function converts the iamge file to 8bit. If the user_format is jpg, this function is skipped.
       Numpy is used for fast vectorization. '''
# Turn image into numbers we can calculate with(np) / Convert the image to a NumPy array for easier numerical processing. Normalize the pixel values to 0255.    
    array = np.array(image_file)
    if array.max() == 0:
        reduced_bit = (array / 1.0)*255
    else:
        reduced_bit = (array / array.max())*255
# Turn the numbers back into an image.        
    img = Image.fromarray(reduced_bit.astype('int8'))
    return img

# Overlay multiple channel images with color to create a single composite color image.
def write_composite(image_files, colors):   
    
    '''Fucntion creates a screen composite image with colors specified by the colors list variable.'''
# Start with the last image file, color it, and use it as the base.    for...:For each other channel-color it and blend it using “screen” mode.  
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
    
    #Loop through all images that match the regex mattern and extract images from the layered 2D morphology files
    print(f'\n...Extracting {user_format} files (raw) from layered 2D morphology image files')   
    for image_file in glob.glob('*.tif') + glob.glob('*.TIF'):
        if regex.match(image_file):
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
            img = force_8bit(img).convert('L') #Calls the function to convert images to 8bit.
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
        image_files = sorted(
    [image_file for image_file in glob.glob(fov + '*')],
    key=lambda x: int(re.search(r'_ch(\d+)_', x).group(1))
)
        composite_image = write_composite(image_files, colors) #Calls the function to create composite images.
        os.chdir(os.path.join(current_dir, composite_folder))
        composite_image.save(f'{fov}_composite.' + user_format, compress_type = compress_value)
        os.chdir(os.path.join(current_dir, composite_autocontrast_folder))
        ImageOps.autocontrast(composite_image, cutoff = clipping, ignore = None, preserve_tone = False ).save(f'{fov}_composite_autocontrast.' + user_format, compress_type = compress_value)
        os.chdir(os.path.join(current_dir, bit_reduced_folder))
               
    end = perf_counter() #Used in calcualting script execution time.
       
    print(f'\nFinished. Total run time = {int(end - start)} sec')