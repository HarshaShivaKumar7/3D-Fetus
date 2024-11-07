import time
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops


# Load the NRRD file
def load_nrrd(file_path):
    # Read the NRRD file
    image = sitk.ReadImage(file_path)

    # Convert the image to a numpy array for easier manipulation
    image_array = sitk.GetArrayFromImage(image)

    return image, image_array


# Simple Thresholding to isolate the fetus based on intensity values
def threshold_fetus(image_array, lower_threshold=50, upper_threshold=200):
    # Apply a simple threshold to isolate the fetus
    binary_image = np.where(
        (image_array > lower_threshold) & (image_array < upper_threshold), 1, 0
    )
    return binary_image


# Display the segmented fetus using thresholding
def display_thresholded_fetus(image_array, thresholded_image):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_array[image_array.shape[0] // 2], cmap="gray")
    axes[0].set_title("Original Slice (Middle)")
    axes[0].axis("off")

    axes[1].imshow(thresholded_image[image_array.shape[0] // 2], cmap="gray")
    axes[1].set_title("Thresholded Fetus (Middle)")
    axes[1].axis("off")

    plt.show()


# Show the basic image information (dimensions, spacing)
def show_image_info(image):
    print("Image Information:")
    print(f"Dimensions: {image.GetDimension()}D")
    print(f"Size: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")


# Play through the thresholded slices
def play_slices(image_array, thresholded_image, interval=0.5):
    for slice_index in range(image_array.shape[0]):
        plt.imshow(image_array[slice_index], cmap="gray")
        plt.title(f"Original Slice {slice_index}")
        plt.axis("off")
        plt.pause(interval)  # Pause for the specified interval (in seconds)
        plt.clf()  # Clear the current plot

        # Thresholded image (fetus area only)
        plt.imshow(thresholded_image[slice_index], cmap="gray")
        plt.title(f"Thresholded Fetus {slice_index}")
        plt.axis("off")
        plt.pause(interval)
        plt.clf()


# ROI


# Simple Thresholding to isolate the fetus based on intensity values
def threshold_fetus(image_array, lower_threshold=50, upper_threshold=200):
    # Apply a simple threshold to isolate the fetus
    binary_image = np.where(
        (image_array > lower_threshold) & (image_array < upper_threshold), 1, 0
    )
    return binary_image


# Automatically crop around the largest connected component (likely the fetus)
def auto_crop_fetus(image_array, thresholded_image):
    # Find connected components in the thresholded image
    labeled_image = label(thresholded_image)

    # Find properties of the labeled regions
    regions = regionprops(labeled_image)

    # Select the largest region (likely the fetus)
    largest_region = max(regions, key=lambda r: r.area)

    # Get the bounding box coordinates of the largest region
    min_row, min_col, min_depth, max_row, max_col, max_depth = largest_region.bbox

    # Crop the image around the bounding box of the largest region
    cropped_image = image_array[min_depth:max_depth, min_row:max_row, min_col:max_col]

    return cropped_image


# Display cropped fetus region
def display_auto_cropped_fetus(image_array, thresholded_image):
    cropped_image = auto_crop_fetus(image_array, thresholded_image)
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    axes.imshow(cropped_image[cropped_image.shape[0] // 2], cmap="gray")
    axes.set_title("Automatically Cropped Fetus Region (Middle Slice)")
    axes.axis("off")
    plt.show()


def manual_crop(image_array, x_start, x_end, y_start, y_end, z_start, z_end):
    cropped_image = image_array[z_start:z_end, y_start:y_end, x_start:x_end]
    return cropped_image


def display_manual_cropped_fetus(
    image_array, x_start, x_end, y_start, y_end, z_start, z_end
):
    cropped_image = manual_crop(
        image_array, x_start, x_end, y_start, y_end, z_start, z_end
    )
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    axes.imshow(cropped_image[cropped_image.shape[0] // 2], cmap="gray")
    axes.set_title("Manually Cropped Fetus Region (Middle Slice)")
    axes.axis("off")
    plt.show()


def main():
    nrrd_file_path = "D:/ultra/nrrd/I0000048.nrrd"

    image, image_array = load_nrrd(nrrd_file_path)

    show_image_info(image)

    thresholded_image = threshold_fetus(image_array)

    display_manual_cropped_fetus(
        image_array, x_start=50, x_end=150, y_start=60, y_end=160, z_start=60, z_end=120
    )

    # Option 2: Display the automatically cropped fetus region (using thresholding and connected components)
    display_auto_cropped_fetus(image_array, thresholded_image)

    # Option 3: Play through the thresholded slices (this is also an option to visualize)
    play_slices(image_array, thresholded_image, interval=0.1)


if __name__ == "__main__":
    main()
