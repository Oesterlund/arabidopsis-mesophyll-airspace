
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters
import os

plt.close('all')

def enhance_contrast(image, method='stretch'):
    """Enhances the contrast of a 2D image using different methods."""
    
    # Convert to float and normalize if needed
    image = image.astype(np.float32)
    
    if image.max() > 1:  # Normalize only if values are not already in range 0-1
        image /= image.max()

    if method == 'stretch':
        # Contrast stretching (Min-Max normalization)
        img_min, img_max = image.min(), image.max()
        enhanced = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    elif method == 'equalize':
        # Histogram Equalization
        enhanced = exposure.equalize_hist(image)
        enhanced = (enhanced * 255).astype(np.uint8)  # Convert back to uint8

    elif method == 'clahe':
        # Adaptive Histogram Equalization (CLAHE)
        enhanced = exposure.equalize_adapthist(image, clip_limit=0.03)
        enhanced = (enhanced * 255).astype(np.uint8)  # Convert back to uint8
        
        # Additional sharpening filter to enhance details
        enhanced = filters.unsharp_mask(enhanced, radius=1.0, amount=1.5)
        enhanced = (enhanced * 255).astype(np.uint8)  # Convert back to uint8

    else:
        raise ValueError("Unknown method. Use 'stretch', 'equalize', or 'clahe'")
    
    return enhanced

def process_tiff(file_path, output_path, method='stretch'):
    """Loads a 2D TIFF file, enhances contrast, displays it, and saves the output."""
    image = tifffile.imread(file_path)

    # Ensure the image is 2D
    if image.ndim != 2:
        raise ValueError("Expected a 2D image.")

    enhanced_image = enhance_contrast(image, method)
    
    # Save the enhanced image as a TIFF file
    tifffile.imwrite(output_path, enhanced_image, photometric='minisblack')
    print(f"Enhanced image saved to {output_path}")
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
     
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title(f'Enhanced ({method})')
    plt.axis('off')
     
    plt.show()
    
    return enhanced_image


saveProccess = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/adaxial/enhanced/'
file_path = "/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/adaxial/"

dataList = os.listdir(file_path)

dataListS = np.sort(dataList)
dataListS1 = dataListS[:-2]

for nameF,m in zip(dataListS1,range(len(dataListS1))):
    print(nameF,m)
    enhancedIm = process_tiff(file_path + nameF, saveProccess + nameF, method='clahe')
    
