# Capstone Project - Image Processing

# NOTE : images in this folder are a small sample so we can upload the file easier. More images were processed for the project

import numpy as np # make sure it's version 1.18.5
import matplotlib.pyplot as plt
import os

# file name requirements
filename_base = 'location_'
location = 34 # change based on location

# get number of images in the filepath for loop
folder_path = 'data/34/'
folder_objects = len(os.listdir(folder_path))
print(folder_objects)

# filepath for cropped images
output_filepath = 'data/34_cropped'
for i in range(1, folder_objects+1):
    filepath = folder_path + '\\' + filename_base + str(location)+ '(' + str(i) + ').jpg'
    print(filepath)

    image = np.array(plt.imread(filepath))
    x, y, z = image.shape
    
    # subtract approx. height of taller watermark
    cropped_height = x - 550 
    cropped_image = image[0:cropped_height, 0:y, :]

    output_image_filepath = output_filepath+'\\'+filename_base + str(location)+ '('+str(i)+').jpg'
    plt.imsave(output_image_filepath, cropped_image)
