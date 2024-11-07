import vtk
import SimpleITK as sitk
import vtk.util.numpy_support as vtk_np
import numpy as np
import os


class InteractiveVolumeRenderer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.image = sitk.ReadImage(self.file_path)
        self.min_intensity, self.max_intensity = self.get_intensity_range(self.image)
        self.image_data = sitk.GetArrayFromImage(self.image)

        self.vtk_image_data = vtk.vtkImageData()
        self.vtk_image_data.SetDimensions(self.image.GetSize()[::-1])
        self.vtk_image_data.SetSpacing(self.image.GetSpacing())
        self.vtk_image_data.GetPointData().SetScalars(
            vtk_np.numpy_to_vtk(self.image_data.ravel(), deep=True)
        )

        self.volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.volume_mapper.SetInputData(self.vtk_image_data)
        self.volume_property = vtk.vtkVolumeProperty()

        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(self.volume_property)
        self.renderer.AddVolume(self.volume)

        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.render_window.SetSize(800, 600)

        self.setup_transfer_functions()
        self.create_sliders()

    def get_intensity_range(self, image):
        array = sitk.GetArrayFromImage(image)
        return np.min(array), np.max(array)

    def create_transfer_function(self):
        scalar_opacity = vtk.vtkPiecewiseFunction()
        scalar_opacity.AddPoint(self.min_intensity, 0.0)
        scalar_opacity.AddPoint(self.max_intensity / 2, 0.6)
        scalar_opacity.AddPoint(self.max_intensity, 0.8)

        gradient_opacity = vtk.vtkPiecewiseFunction()
        gradient_opacity.AddPoint(self.min_intensity, 0.0)
        gradient_opacity.AddPoint(self.max_intensity / 4, 1.0)
        gradient_opacity.AddPoint(self.max_intensity / 2, 0.8)
        gradient_opacity.AddPoint(self.max_intensity, 0.4)

        color_transfer = vtk.vtkColorTransferFunction()
        color_transfer.AddRGBPoint(self.min_intensity, 0.54902, 0.25098, 0.14902)
        color_transfer.AddRGBPoint(self.max_intensity / 2, 0.882353, 0.603922, 0.290196)
        color_transfer.AddRGBPoint(self.max_intensity, 0.694, 0.478, 0.396)

        return scalar_opacity, gradient_opacity, color_transfer

    def setup_transfer_functions(self):
        scalar_opacity, gradient_opacity, color_transfer = (
            self.create_transfer_function()
        )
        self.volume_property.SetScalarOpacity(scalar_opacity)
        self.volume_property.SetGradientOpacity(gradient_opacity)
        self.volume_property.SetColor(color_transfer)
        self.volume_property.SetDiffuse(0.9)
        self.volume_property.SetSpecular(0.2)
        self.volume_property.SetSpecularPower(10)
        self.volume_property.SetAmbient(0.1)
        self.volume_property.ShadeOn()

    def create_slider(self, min_value, max_value, initial_value, label):
        slider_rep = vtk.vtkSliderRepresentation2D()
        slider_rep.SetMinimumValue(min_value)
        slider_rep.SetMaximumValue(max_value)
        slider_rep.SetValue(initial_value)
        slider_rep.SetTitleText(label)
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(10, 30)
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(200, 30)
        slider_rep.SetSliderLength(0.02)
        slider_rep.SetSliderWidth(0.02)
        slider_rep.SetEndCapWidth(10)

        slider_widget = vtk.vtkSliderWidget()
        slider_widget.SetInteractor(self.render_window_interactor)
        slider_widget.SetRepresentation(slider_rep)
        slider_widget.SetAnimationModeToAnimate()
        return slider_widget, slider_rep

    def create_sliders(self):
        self.opacity_slider_widget, self.opacity_slider_rep = self.create_slider(
            0.0, 1.0, 0.6, "Scalar Opacity"
        )
        self.opacity_slider_widget.AddObserver(
            "InteractionEvent", self.update_scalar_opacity
        )

        self.gradient_slider_widget, self.gradient_slider_rep = self.create_slider(
            0.0, 1.0, 0.8, "Gradient Opacity"
        )
        self.gradient_slider_widget.AddObserver(
            "InteractionEvent", self.update_gradient_opacity
        )

        self.color_slider_r_widget, self.color_slider_r_rep = self.create_slider(
            0.0, 1.0, 0.54902, "Red"
        )
        self.color_slider_g_widget, self.color_slider_g_rep = self.create_slider(
            0.0, 1.0, 0.25098, "Green"
        )
        self.color_slider_b_widget, self.color_slider_b_rep = self.create_slider(
            0.0, 1.0, 0.14902, "Blue"
        )

        self.color_slider_r_widget.AddObserver("InteractionEvent", self.update_color)
        self.color_slider_g_widget.AddObserver("InteractionEvent", self.update_color)
        self.color_slider_b_widget.AddObserver("InteractionEvent", self.update_color)

    def update_scalar_opacity(self, obj, event):
        opacity_value = self.opacity_slider_rep.GetValue()
        scalar_opacity = self.volume_property.GetScalarOpacity()
        scalar_opacity.RemoveAllPoints()
        scalar_opacity.AddPoint(self.min_intensity, 0.0)
        scalar_opacity.AddPoint(self.max_intensity / 2, opacity_value)
        scalar_opacity.AddPoint(self.max_intensity, opacity_value)
        self.render_window.Render()

    def update_gradient_opacity(self, obj, event):
        gradient_value = self.gradient_slider_rep.GetValue()
        gradient_opacity = self.volume_property.GetGradientOpacity()
        gradient_opacity.RemoveAllPoints()
        gradient_opacity.AddPoint(self.min_intensity, 0.0)
        gradient_opacity.AddPoint(self.max_intensity / 4, gradient_value)
        gradient_opacity.AddPoint(self.max_intensity / 2, gradient_value)
        gradient_opacity.AddPoint(self.max_intensity, gradient_value)
        self.render_window.Render()

    def update_color(self, obj, event):
        r_value = self.color_slider_r_rep.GetValue()
        g_value = self.color_slider_g_rep.GetValue()
        b_value = self.color_slider_b_rep.GetValue()

        color_transfer = self.volume_property.GetColor()
        color_transfer.RemoveAllPoints()
        color_transfer.AddRGBPoint(self.min_intensity, r_value, g_value, b_value)
        color_transfer.AddRGBPoint(self.max_intensity / 2, 0.882353, 0.603922, 0.290196)
        color_transfer.AddRGBPoint(self.max_intensity, 0.694, 0.478, 0.396)
        self.render_window.Render()

    def render(self):
        self.render_window.Render()
        self.render_window_interactor.Start()


def process_all_nrrd_files(directory):
    nrrd_files = [f for f in os.listdir(directory) if f.endswith(".nrrd")]

    for file_name in nrrd_files:
        file_path = os.path.join(directory, file_name)
        print(f"Processing file: {file_path}")
        renderer = InteractiveVolumeRenderer(file_path)
        renderer.render()


def main():
    directory = r"D:/Ultra/nrrd"

    process_all_nrrd_files(directory)


if __name__ == "__main__":
    main()
