import skimage
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import label   
import scipy.stats
import scipy.ndimage as ndimage
from scipy.ndimage import label, sum as nd_sum
import scipy.ndimage as ndi
from skimage.measure import moments_central, inertia_tensor
import re
import os
from skimage.io import imread
import pandas as pd
from scipy.stats import bootstrap
import seaborn as sns
from statannotations.Annotator import Annotator
import scienceplots
from scipy.stats import shapiro
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

plt.style.use(['science','bright']) # sans-serif font
plt.close('all')

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.rc('axes', labelsize=12)

params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize':12,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
plt.rcParams.update(params)


cmap=sns.color_palette("colorblind")

pixel=1.3

###############################################################################
#
#
#
###############################################################################

path = "/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/adaxial/segmentations/"
pathImg = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/adaxial/enhanced/'

dataList = os.listdir(path)

dataListS = np.sort(dataList)

cropping_dict = {
    "156_Col0_w6_p1_l7t_zoomed-0.25.tif": (0, None, 0, 300),
    "131_ROP_w6_p1_l8m_zoomed-0.25.tif": (0, None, 0, 400),
    "136_RIC_w6_p2_l7b_zoomed-0.25.tif": (120, None,0, None),
    "145_RIC_w6_p1_l6t_zoomed-0.25.tif": (270, None, 0,  None)
}

# Define images to exclude from specific calculations
excluded_images = {"157_Col0_w6_p2_l6b_zoomed-0.25.tif"}

df_results = []
for nameF in dataListS:
    group = re.search(r'_(Col0|RIC|ROP)_', nameF)
    group = group.group(1) if group else "Unknown"

    img = imread(path+nameF)
    
    # Calculate areas on the original image
    imgD = skimage.morphology.dilation(img)
    labeled_array, num_features = label(imgD)
    areas = nd_sum(imgD, labeled_array, index=range(1, num_features + 1))
    
    # Store area calculations (always based on the full image)
    for i, area in enumerate(areas, start=1):
        df_results.append({
            "Group": group,
            "Filename": nameF,
            "Region_ID": i,
            "Area_Pixels": area,
            "Area_um2": area * (pixel ** 2)  # Convert to µm²
        })
        
    # Skip density calculation for excluded images
    if nameF not in excluded_images:
        # If the image is in cropping_dict, apply cropping before calculating density
        if nameF in cropping_dict:
            y_min, y_max, x_min, x_max = cropping_dict[nameF]
            img_cropped = img[y_min:y_max, x_min:x_max]  # Crop image for density
        else:
            img_cropped = img  # No cropping needed
        
        # Compute density on cropped image
        imgD_cropped = skimage.morphology.dilation(img_cropped)
        labeled_array_cropped, num_features_cropped = label(imgD_cropped)
        total_pixels_cropped = img_cropped.size
        density = num_features_cropped / total_pixels_cropped if total_pixels_cropped > 0 else 0  # Avoid division by zero

        # Store density results
        df_results.append({
            "Group": group,
            "Filename": nameF,
            "Region_ID": "Density",
            "Area_Pixels": None,  # Not relevant for density
            "Area_um2": None,  # Not relevant for density
            "Density": density * 1e6  # Convert μm² to mm²
        })




df_results = pd.DataFrame(df_results)
df_results["Density"] = df_results["Density"].astype(float)  # Convert to float to allow NaN
df_results["Density"].fillna(np.nan, inplace=True) 

df_results.to_csv("/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/scripts/adaxial_annotated_areas_with_groups.csv", index=False)

grouped_means = df_results.groupby("Group")["Area_um2"].mean().reset_index()
dens_means = df_results.groupby("Group")["Density"].mean().reset_index()


lisT = ['Col0','RIC','ROP']
for group in lisT:
    
    group_data = df_results[df_results['Group'] == group]
    
    per_image_means = group_data.groupby('Filename')['Area_um2'].mean().dropna().values
    
    res = bootstrap((per_image_means,), np.mean, n_resamples=10000, confidence_level=0.90)
    
    # Step 4: Extract confidence interval
    ci_lower, ci_upper = res.confidence_interval
    
    # Step 5: Print results
    print(group, f"{np.mean(per_image_means):.1f} 90% confidence interval for the mean: [{ci_lower:.1f}, {ci_upper:.1f}]")
    '''
    data = np.asarray(df_results[df_results['Group']==lisT[i]]['Area_um2'].dropna())

    res = bootstrap((data,), np.mean, n_resamples=10000,confidence_level=0.90)
    
    # The bootstrap confidence interval
    ci_lower, ci_upper = res.confidence_interval
    
    print(lisT[i],f"{np.mean(data):.1f} 90% confidence interval for the mean: [{ci_lower:.1f}, {ci_upper:.1f}]")
    '''

for i in range(3):
    data = np.asarray(df_results[df_results['Group']==lisT[i]]['Density'].dropna())

    res = bootstrap((data,), np.mean, n_resamples=10000,confidence_level=0.90)
    
    # The bootstrap confidence interval
    ci_lower, ci_upper = res.confidence_interval
    
    print(lisT[i],f"{np.mean(data):.2f} 90% confidence interval for the mean: [{ci_lower:.2f}, {ci_upper:.2f}]")

    
###############################################################################
#
# abaxial
#
print('data for abaxial side')
pathImg = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/abaxial/enhanced/'
dataList = os.listdir(path)

excluded_images = {}
df_resultsAba = []
for nameF in dataListS:

    group = re.search(r'_(Col0|RIC|ROP)_', nameF)
    group = group.group(1) if group else "Unknown"

    img = imread(path+nameF)
    
    # Calculate areas on the original image
    imgD = skimage.morphology.dilation(img)
    labeled_array, num_features = label(imgD)
    areas = nd_sum(imgD, labeled_array, index=range(1, num_features + 1))
    
    
        
    
    # Compute density on cropped image
    imgD_cropped = skimage.morphology.dilation(img)
    labeled_array_cropped, num_features_cropped = label(imgD_cropped)
    total_pixels_cropped = img_cropped.size
    density = num_features_cropped / total_pixels_cropped if total_pixels_cropped > 0 else 0  # Avoid division by zero

    # Store area calculations (always based on the full image)
    for i, area in enumerate(areas, start=1):
        df_resultsAba.append({
            "Group": group,
            "Filename": nameF,
            "Region_ID": i,
            "Area_Pixels": area,
            "Area_um2": area * (pixel ** 2),  # Convert to µm²
            "Density": density * 1e6
        })



df_resultsAba = pd.DataFrame(df_resultsAba)
df_resultsAba["Density"] = df_resultsAba["Density"].astype(float)  # Convert to float to allow NaN
df_resultsAba["Density"].fillna(np.nan, inplace=True) 

df_resultsAba.to_csv("/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/scripts/adaxial_annotated_areas_with_groups.csv", index=False)

grouped_means = df_resultsAba.groupby("Group")["Area_um2"].mean().reset_index()
dens_means = df_resultsAba.groupby("Group")["Density"].mean().reset_index()


lisT = ['Col0','RIC','ROP']

for group in lisT:
    group_data = df_resultsAba[df_resultsAba['Group'] == group]
    
    per_image_means = group_data.groupby('Filename')['Density'].mean().dropna().values

    res = bootstrap((per_image_means,), np.mean, n_resamples=10000,confidence_level=0.90)
    
    # The bootstrap confidence interval
    ci_lower, ci_upper = res.confidence_interval
    
    print(group,f"{np.mean(per_image_means)} 90% confidence interval for the mean: [{ci_lower}, {ci_upper}]")
 
    
###############################################################################
#
# statistical comparisons
#
###############################################################################

############
# firstly we have to create a mean per leaf representation, or we get type I error
df_leaf_means = df_resultsAba.groupby(['Filename', 'Group'])['Density'].mean().reset_index()


###############################################################################
# density abaxial

####
# check for normality in distributions

subset_pred = df_leaf_means[['Group', 'Density']]

wtData = subset_pred['Density'][subset_pred['Group']=='Col0']
ricData = subset_pred['Density'][subset_pred['Group']=='RIC']
ropData = subset_pred['Density'][subset_pred['Group']=='ROP']
# Run the Shapiro-Wilk test
stat, p_valueWT = shapiro(wtData)
stat, p_valueRIC = shapiro(ricData)
stat, p_valueROP = shapiro(ropData)

#  normal distributed, do anova

f_oneway(wtData, ricData, ropData)

tukey = pairwise_tukeyhsd(endog=subset_pred['Density'], groups=subset_pred['Group'], alpha=0.05)
print(tukey)

    

###############################################################################
# density

####
# check for normality in distributions

df_leaf_meansAda = df_results.groupby(['Filename','Group'])[['Density','Area_um2']].mean().dropna().reset_index()


subset_predAd = df_leaf_meansAda[['Group', 'Density']]

wtData = subset_predAd['Density'][subset_predAd['Group']=='Col0']
ricData = subset_predAd['Density'][subset_predAd['Group']=='RIC']
ropData = subset_predAd['Density'][subset_predAd['Group']=='ROP']
# Run the Shapiro-Wilk test
stat, p_valueWT = shapiro(wtData)
stat, p_valueRIC = shapiro(ricData)
stat, p_valueROP = shapiro(ropData)

#  normal distributed, do anova

f_oneway(wtData, ricData, ropData)

tukey = pairwise_tukeyhsd(endog=subset_predAd['Density'], groups=subset_predAd['Group'], alpha=0.05)
print(tukey)
