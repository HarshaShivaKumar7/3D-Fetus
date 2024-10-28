import SimpleITK as sitk
import nrrd
import numpy as np
import matplotlib.pyplot as plt

# Load the NRRD file
nrrd_file_path = 'D:/ultra/nrrd/I0000048.nrrd'
image, header = nrrd.read(nrrd_file_path)

# Convert the numpy array to a SimpleITK Image
sitk_image = sitk.GetImageFromArray(image)

# Step 1: Apply Gaussian smoothing to reduce noise
smoothed_image = sitk.SmoothingRecursiveGaussian(sitk_image, sigma=2.0)

# Step 2: Use histogram equalization to enhance contrast
equalized_image = sitk.AdaptiveHistogramEqualization(smoothed_image)

# Step 3: Use Otsu's thresholding to separate the fetus from the background
otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.SetInsideValue(0)  # Background
otsu_filter.SetOutsideValue(1)  # Foreground (fetus)
binary_image = otsu_filter.Execute(equalized_image)

# Step 4: Remove small connected components (artifacts)
connected_component_filter = sitk.ConnectedComponentImageFilter()
binary_cc = connected_component_filter.Execute(binary_image)

# Keep only the largest connected component (assuming it's the fetus)
label_shape_statistics = sitk.LabelShapeStatisticsImageFilter()
label_shape_statistics.Execute(binary_cc)
largest_label = max((label_shape_statistics.GetPhysicalSize(l), l) for l in label_shape_statistics.GetLabels())[1]
fetus_mask = sitk.BinaryThreshold(binary_cc, largest_label, largest_label, 1, 0)

# Convert the mask back to a numpy array for visualization
fetus_mask_array = sitk.GetArrayFromImage(fetus_mask)

# Display the original and segmented images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image[image.shape[0] // 2], cmap="gray")  # Display a central slice
axes[0].set_title("Original Ultrasound Slice")
axes[1].imshow(fetus_mask_array[fetus_mask_array.shape[0] // 2], cmap="gray")  # Display the segmented fetus slice
axes[1].set_title("Segmented Fetus")
plt.show()
