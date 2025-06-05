import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters
import os

plt.close('all')

def enhance_contrast(image, method='stretch'):
    """Enhances the contrast of a 3D image using different methods."""
    if method == 'stretch':
        # Contrast stretching (Min-Max normalization)
        img_min, img_max = image.min(), image.max()
        enhanced = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    elif method == 'equalize':
        # Histogram Equalization
        enhanced = np.array([exposure.equalize_hist(slice_) for slice_ in image])
        enhanced = (enhanced * 255).astype(np.uint8)  # Convert back to uint8
    elif method == 'clahe':
        # Adaptive Histogram Equalization (CLAHE)
        enhanced = np.array([exposure.equalize_adapthist(slice_, clip_limit=0.03) for slice_ in image])
        enhanced = (enhanced * 255).astype(np.uint8)  # Convert back to uint8
        
        # Additional sharpening filter to enhance details
        enhanced = np.array([filters.unsharp_mask(slice_, radius=1.0, amount=1.5) for slice_ in enhanced])
        enhanced = (enhanced * 255).astype(np.uint8)  # Convert back to uint8
    else:
        raise ValueError("Unknown method. Use 'stretch', 'equalize', or 'clahe'")
    
    return enhanced

def process_tiff(file_path, output_path, method='stretch'):
    """Loads a 3D TIFF file, enhances contrast, displays it, and saves the output."""
    image = tifffile.imread(file_path)
    
    if image.ndim != 3:
        raise ValueError("Expected a 3D image stack.")
    
    enhanced_image = enhance_contrast(image, method)
    
    # Save the enhanced image as a TIFF file
    tifffile.imwrite(output_path, enhanced_image, photometric='minisblack')
    print(f"Enhanced image saved to {output_path}")
    
    # Display a middle slice for visualization
    slice_index = image.shape[0] // 2
    '''
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image[slice_index], cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image[slice_index], cmap='gray')
    plt.title(f'Enhanced ({method})')
    plt.axis('off')
    
    plt.show()
    '''
    return enhanced_image

saveProccess = '/home/isabella/Documents/PLEN/x-ray/ali_data/stomata/Ali_Leverett/processed_images/'
file_path = "/home/isabella/Documents/PLEN/x-ray/ali_data/stomata/Ali_Leverett/data/"

dataList = os.listdir(file_path)

dataListS = np.sort(dataList)
dataListS1 = dataListS[1:]

for nameF,m in zip(dataListS1,range(len(dataListS1))):
    print(nameF,m)
    enhancedIm = process_tiff(file_path + nameF, saveProccess + nameF, method='clahe')
    
