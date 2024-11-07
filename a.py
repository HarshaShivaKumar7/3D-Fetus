import time
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


# Load the NRRD file
def load_nrrd(file_path):
    # Read the NRRD file
    image = sitk.ReadImage(file_path)

    # Convert the image to a numpy array for easier manipulation
    image_array = sitk.GetArrayFromImage(image)

    return image, image_array


# Display multiple slices in a grid
def display_multiple_slices(image_array, start_slice, num_slices_to_show=9):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            slice_index = start_slice + i * 3 + j
            if slice_index < image_array.shape[0]:
                axes[i, j].imshow(image_array[slice_index], cmap="gray")
                axes[i, j].set_title(f"Slice {slice_index}")
                axes[i, j].axis("off")  # Hide axes
            else:
                axes[i, j].axis("off")  # Hide empty subplots
    plt.show()


# Show the basic image information (dimensions, spacing)
def show_image_info(image):
    print("Image Information:")
    print(f"Dimensions: {image.GetDimension()}D")
    print(f"Size: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")


# Crop to a region of interest (center crop)
def crop_center(image_array, crop_size=(100, 100)):
    z_center, y_center, x_center = (
        image_array.shape[0] // 2,
        image_array.shape[1] // 2,
        image_array.shape[2] // 2,
    )
    z_half, y_half, x_half = crop_size[0] // 2, crop_size[1] // 2, crop_size[1] // 2
    cropped_image = image_array[
        z_center - z_half : z_center + z_half,
        y_center - y_half : y_center + y_half,
        x_center - x_half : x_center + x_half,
    ]
    return cropped_image


# Display the cropped region
def display_cropped_slices(image_array, crop_size=(100, 100)):
    cropped_image = crop_center(image_array, crop_size)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            slice_index = i * 3 + j
            if slice_index < cropped_image.shape[0]:
                axes[i, j].imshow(cropped_image[slice_index], cmap="gray")
                axes[i, j].set_title(f"Slice {slice_index}")
                axes[i, j].axis("off")  # Hide axes
            else:
                axes[i, j].axis("off")
    plt.show()


# In main(), use `display_cropped_slices()` to zoom in


def play_slices(image_array, interval=0.5):
    for slice_index in range(image_array.shape[0]):
        plt.imshow(image_array[slice_index], cmap="gray")
        plt.title(f"Slice {slice_index}")
        plt.axis("off")
        plt.pause(interval)  # Pause for the specified interval (in seconds)
        plt.clf()  # Clear the current plot


# Call the play_slices() function in your main loop
def main():
    # Replace this with the path to your NRRD file
    nrrd_file_path = "D:/ultra/nrrd/I0000048.nrrd"

    # Load the NRRD file
    image, image_array = load_nrrd(nrrd_file_path)

    # Show image information (dimensions, spacing, etc.)
    show_image_info(image)

    # Play through all the slices
    play_slices(image_array, interval=0.1)


if __name__ == "__main__":
    main()
