# SlicerSandbox
![Logo](Sandbox_Logo_128.png)

Collection of modules for 3D Slicer, which are already useful, but not finalized, polished, or proven to be useful enough to be included in Slicer core.

## Lights

This module can be used to adjust lighting and rendering options in 3D views. Select all or specific 3D views at the top, then adjust options in sections below.

- Lighting: configures a [lightkit](https://vtk.org/doc/nightly/html/classvtkLightKit.html) that is used for rendering of all 3D content, including volume rendering. The kit consists of the key light (typically the strongest light, simulating overhead lighting, such as ceiling lights or sun), fill light (typically same side as key light, but other side; simulating diffuse reflection of key light), headlight (moves with the camera, reduces contrast between key light and fill light), back lights (fill on the high-contrast areas behind the object). 
- Ambient shadows: Uses screen space ambient occlusion (SSAO) method to simulate shadows. Size scale determines the details that are emphasized. The scale is logarithmic, the default 0 value corresponds to 100mm. For highlighting smaller details (such as uneven surface), reduce the value. Use larger values to make large objects better distinguishable. These settings have no effect on volume rendering.
- Image-based lighting: Necessary when models are displayed with PBR (physics based rendering) interpolation. Brightness of the image determines the amount of light reflected from object surfaces; and fragments of the image appears as reflection on surface of smooth metallic objects. Currently only a single picture is provided via the user interface ([hospital_room](https://github.com/PerkLab/SlicerSandbox/blob/master/Lights/Resources/hospital_room.jpg)), but other images can be downloaded (for example from [polyhaven.com](https://polyhaven.com)) and be used in the Python API of the module. These settings have no effect on volume rendering.

## Curved Planar Reformat

Curved planar reformat module allows "straightening" a curved volume for visualization or quantification. The module provides two-way spatial mapping between the original and straightened space.

### Adjust reformatting parameters for robust mapping

If the slice size is too large or curve resolution is too fine then in some regions you can have transform that maps the same point into different positions (the displacement field folds into itself). In these regions the transforms in not invertible.

To reduce these ambiguously mapped regions, decrease `Slice size`. If necessary `Curve resolution` can be slightly increased as well (it controls how densely the curve is sampled to generate the displacement field, if samples are farther from each other then it may reduce chance of contradicting samples).

![image|561x379](https://aws1.discourse-cdn.com/standard17/uploads/slicer/original/2X/3/3c6ee214e10415a6eb1fb53638c02e44bc93d4a1.png)

You can quickly validate the transform, by going to Transforms module and in Display/Visualization section check all 3 checkboxes, the straightened image as `Region` and visualization mode to `Grid`.

For example, this transform results in a smooth displacement field, it is invertible in the visualized region:

![image|690x420](https://aws1.discourse-cdn.com/standard17/uploads/slicer/optimized/2X/0/045048696d683fffafbea801edeb05cc1349abd5_2_1035x630.png)

If the slice size is increased then folding occurs:

![image|489x500](https://aws1.discourse-cdn.com/standard17/uploads/slicer/optimized/2X/e/e6c2b020c07ed52c5b37b665bf424d9e82738cd0_2_733x750.png)

Probably you can find a parameter set that works for a large group of patients. Maybe one parameter set works for all, but maybe you need to have a few different presets (small, medium, large)
