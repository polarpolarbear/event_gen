import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


import cv2
import numpy as np

def resize_with_padding(image_path, output_size):
    # Read the image
    image = cv2.imread(image_path)
    
    # Compute aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    
    # Calculate the new size while preserving aspect ratio
    if aspect_ratio > 1:  # landscape orientation
        new_width = output_size
        new_height = round(output_size / aspect_ratio)
    else:  # portrait orientation
        new_width = round(output_size * aspect_ratio)
        new_height = output_size
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Create a new blank image with the desired output size
    padded_image = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    
    # Calculate position to paste resized image with zero padding
    x_offset = (output_size - new_width) // 2
    y_offset = (output_size - new_height) // 2
    
    # Paste resized image onto the blank image with zero padding
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    
    return padded_image

# Example usage:
image_path = "/root/event_gen-main/data/Caltech101/ant/image_0014.jpg"
output_size = 240  # desired output size
padded_image = resize_with_padding(image_path, output_size)
plt.imshow(padded_image, cmap='magma')
plt.savefig('tmp.jpg')

# Now padded_image contains the resized image with zero padding as a numpy array
