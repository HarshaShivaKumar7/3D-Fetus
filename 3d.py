import SimpleITK as sitk
import numpy as np
from mayavi import mlab


# Load the NRRD file
def load_nrrd(file_path):
    # Read the NRRD file using SimpleITK
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)  # Convert to numpy array
    return image, image_array


# Simple Thresholding to isolate the fetus (based on intensity)
def threshold_fetus(image_array, lower_threshold=50, upper_threshold=200):
    # Apply a simple threshold to isolate the fetus from other tissues
    binary_image = np.where(
        (image_array > lower_threshold) & (image_array < upper_threshold), 1, 0
    )
    return binary_image


# Visualize the 3D volume using Mayavi
def visualize_3d(image_array):
    # Flip image array if needed to match the correct orientation
    image_array = np.flip(image_array, axis=0)

    # Create a 3D plot with Mayavi
    mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))

    # Visualize the thresholded 3D volume (contour plot)
    mlab.contour3d(
        image_array, contours=10, opacity=0.3, color=(0, 1, 0)
    )  # Green contours

    # Show the 3D plot
    mlab.show()


# Main function to load, threshold and visualize the fetus
def main():
    # Path to your NRRD file (replace with the actual file path)
    nrrd_file_path = "D:/ultra/nrrd/I0000048.nrrd"
    # Load the NRRD file
    image, image_array = load_nrrd(nrrd_file_path)

    # Apply thresholding to isolate the fetus (you can adjust thresholds based on your data)
    thresholded_image = threshold_fetus(image_array)

    # Visualize the thresholded 3D fetus data
    visualize_3d(thresholded_image)


if __name__ == "__main__":
    main()
