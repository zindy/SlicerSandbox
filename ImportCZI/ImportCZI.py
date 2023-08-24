# vim: set tabstop=2 softtabstop=2 shiftwidth=2 expandtab

import os
import unittest
import logging
import vtk, qt, ctk, slicer
import slicer
from slicer.ScriptedLoadableModule import *

import SimpleITK as sitk
import sitkUtils
from multiprocessing import cpu_count

from pathlib import Path

#
# ImportCZI
#

class ImportCZI(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ImportCZI"
    self.parent.categories = ["Informatics"]
    self.parent.dependencies = []
    self.parent.contributors = ["Egor Zindy (Université Libre Bruxelles), Andras Lasso (PerkLab)"]
    self.parent.helpText = """Load CZI image volumes"""
    self.parent.acknowledgementText = """This file borrows heavily from ImportOCT, which was originally developed by Andras Lasso, PerkLab."""
    # don't show this module - it is only for registering a reader
    parent.hidden = True 

#
# Reader plugin
# (identified by its special name <moduleName>FileReader)
#

class ImportCZIFileReader(object):

  def __init__(self, parent):
    self.parent = parent

  def description(self):
    return 'CZI image'

  def fileType(self):
    return 'CZIImageFile'

  def extensions(self):
    # TODO: we could expose all file formats that oct-converter Python package supports,
    # but we would need sample data sets to test with
    return ['Zeiss CZI image file (*.czi)']

  def checkRequiredPythonPackages(self):
    try:
      from czitools import read_tools
      # Successfully imported
      return True
    except ModuleNotFoundError as e:
      message = "Importing of CZI files require the installation of a forked 'czitools' Python package from github (https://github.com/zindy/czitools_lite)."
      from shutil import which
      if which("git") is None:
        slicer.util.errorDisplay(message+"\n\nUnfortunately, 3D slicer cannot access the 'git' command.\nPlease make sure it is installed or accessible via the system path.")
        return False

      if slicer.util.confirmOkCancelDisplay(message+"\n\nClick OK to install it now (it may take several minutes)."):
        # Install converter
        slicer.util.pip_install('git+https://github.com/zindy/czitools_lite.git@main')
      else:
        # User chose not to install the napari package
        return False

    # Failed once, but may have been installed successfully since then, test it now
    try:
      from czitools import read_tools
    except ModuleNotFoundError as e:
      slicer.util.errorDisplay("Required 'czitools' Python package has not been installed. Cannot import CZI image.")
      return False

    return True

  def canLoadFile(self, filePath):
    if not self.checkRequiredPythonPackages():
      return False

    try:
      from czitools import read_tools
      from czitools import metadata_tools as czimd
      # get the complete metadata at once as one big class
      mdata = czimd.CziMetadata(filePath)
    except Exception as e:
      return False
    return True

  def load(self, properties):
    if not self.checkRequiredPythonPackages():
      return False

    try:
      filePath = properties['fileName']
      from czitools import read_tools
      from czitools import metadata_tools as czimd

      import numpy as np

      array6d, mdata, dim_string6d = read_tools.read_6darray(filePath,
        output_order="STCZYX",
        use_dask=True,
        chunk_zyx=True,
        S=0,
        # T=0,
        # Z=0
      )

      mdict = czimd.create_mdict_red(mdata, sort=False)
      n_scenes, n_tps, n_channels, n_zs, n_ys, n_xs = array6d.shape

      #From here we should be able to get the data...

      # Here we can only deal with scene 0, but ideally, would need to ask the user to choose
      # a scene (possibly showing a drop down menu of the scene names?)
      #scene_data = result[0]

      # FIXME assumes units are always returned by the Napari reader in micrometers (Slicer units are mm internally).
      #scale_xyzt = [l/1000. for l in scene_data[1]['scale']][::-1]
      #translate_xyzt = [l/1000. for l in scene_data[1]['translate']][::-1]

      # The numpy array
      #array_tzcyx = scene_data[0]

      #Save all the timepoints
      for t in range(n_tps):
        for c in range(n_channels):
          # Get node base name from filename
          if 'name' in properties.keys():
            base_name = properties['name']
          else:
            base_name =  f"{Path(filePath).name}"
            if n_tps > 1:
              base_name += f" - tp{t}"
            base_name += f" - {mdict['ChannelNames'][c]}"

          base_name = slicer.mrmlScene.GenerateUniqueName(base_name)

          volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', base_name)

          # FIXME assumes units are always returned by the Napari reader in micrometers (Slicer units are mm internally).
          volumeNode.SetSpacing(mdict['XScale']/1000., mdict['YScale']/1000., mdict['ZScale']/1000.)
          #volumeNode.SetOrigin(*translate_xyzt[:-1])

          voxels = array6d[0,t,c,:,:,:]
          slicer.util.updateVolumeFromArray(volumeNode, voxels)
          slicer.util.setSliceViewerLayers(volumeNode, fit=True)

          #TODO This is where we set the max color
          #scalarRange = meta['contrast_limits']
          scalarRange = [np.min(voxels), np.max(voxels)]

          vxr = np.resize(voxels, [d//4 if d>4 else d for d in voxels.shape])
          midPoint = np.quantile(vxr,.999)

          black=[0.,0.,0.]  
          white = None
          if 'ChannelColors' in mdict.keys():
            # Convert #AARRGGBB into 0-1 (r,g,b) tuple
            color = mdict['ChannelColors'][c]
            if color is not None and len(color) == 9 and color[0] == '#':
              # source https://stackoverflow.com/a/29643643
              white = [int(color[i:i+2], 16)/255 for i in (3, 5, 7)]

          if white is None:
            white=[1.,1.,1.]

          colorTransfer= self.createColorTransfer(black, white, scalarRange, midPoint)
          volumePropertyNode=self.createVolumePropertyNode(colorTransfer, scalarRange)

          vrLogic = slicer.modules.volumerendering.logic()
          vrDisplayNode = vrLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
          vrDisplayNode.SetVisibility(True)
          # Use the same window/level and colormap settings for volume rendering as for slice display
          # vrDisplayNode.SetFollowVolumeDisplayNode(True)

          vrDisplayNode.SetAndObserveVolumePropertyNodeID(volumePropertyNode.GetID())
          # Recenter 3D view
          slicer.util.resetThreeDViews()

    except Exception as e:
      logging.error('Failed to load file: '+str(e))
      import traceback
      traceback.print_exc()
      return False

    # Show volume
    selectionNode = slicer.app.applicationLogic().GetSelectionNode()
    selectionNode.SetActiveVolumeID(volumeNode.GetID())
    slicer.app.applicationLogic().PropagateVolumeSelection()

    self.parent.loadedNodes = [volumeNode.GetID()]
    return True
  
  # Code shamelessly lifted from https://github.com/GuillermoCarbajal/slicelets/blob/master/USGuidedProcedure/USGuidedProcedure.py
  def createColorTransfer(self, black=[0.,0.,0.], white=[1.,1.,1.], scalarRange=[0.,65535.], midPoint=None):
    # the color function is configured
    # zero is associated to the scalar zero and 1 to the scalar 255
    colorTransfer = vtk.vtkColorTransferFunction()
    colorTransfer.AddRGBPoint(scalarRange[0], black[0], black[1], black[2])
    if midPoint is not None:
        colorTransfer.AddRGBPoint(midPoint, white[0], white[1], white[2]);
    colorTransfer.AddRGBPoint(scalarRange[1], white[0], white[1], white[2]);
    return colorTransfer
  
  def createVolumePropertyNode(self, colorTransfer, scalarRange=[0.,65535.], windowLevelMinMax=[0.,65535.]):
    volumePropertyNode = slicer.vtkMRMLVolumePropertyNode()
    slicer.mrmlScene.RegisterNodeClass(volumePropertyNode);

    # the scalar opacity mapping function is configured
    # it is a ramp with opacity of 0 equal to zero and opacity of 1 equal to 1. 
    scalarOpacity = vtk.vtkPiecewiseFunction()
    scalarOpacity.AddPoint(scalarRange[0], 0.)
    #scalarOpacity.AddPoint(windowLevelMinMax[0], 0.)
    #scalarOpacity.AddPoint(windowLevelMinMax[1], 1.)
    scalarOpacity.AddPoint(scalarRange[1], 1.)
    volumePropertyNode.SetScalarOpacity(scalarOpacity);
    
    vtkVolumeProperty = volumePropertyNode.GetVolumeProperty()
    
    vtkVolumeProperty.SetInterpolationTypeToNearest();
    vtkVolumeProperty.ShadeOn();
    vtkVolumeProperty.SetAmbient(0.30);
    vtkVolumeProperty.SetDiffuse(0.60);
    vtkVolumeProperty.SetSpecular(0.50);
    vtkVolumeProperty.SetSpecularPower(40);
    
    volumePropertyNode.SetColor(colorTransfer)
   
    slicer.mrmlScene.AddNode(volumePropertyNode)
    
    return volumePropertyNode

#
# ImportCZITest
#

class ImportCZITest(ScriptedLoadableModuleTest):

  def setUp(self):
    slicer.mrmlScene.Clear()

  def runTest(self):
    self.setUp()
    self.test_ImportCZI1()

  def test_ImportCZI1(self):

    self.delayDisplay("Loading test image as segmentation")

    import SampleData
    testFdaFilePath = SampleData.downloadFromURL(
      fileNames='FIXME.czi',
      uris='https://github.com/PerkLab/SlicerSandbox/releases/download/TestingData/FIXME.czi',
      checksums='SHA256:5536006a2bb4d117f5804e49d393ecc1cb7e98c3e5f7924b7b83b8f0e0567e2c')[0]
    volumeNode = slicer.util.loadNodeFromFile(testFdaFilePath, 'CZIImageFile')
    self.assertIsNotNone(volumeNode)

    # FIXME -- Find an image, check these things...
    self.delayDisplay('Checking loaded image')
    self.assertEqual(volumeNode.GetImageData().GetDimensions()[0], 9)
    self.assertEqual(volumeNode.GetImageData().GetDimensions()[1], 320)
    self.assertEqual(volumeNode.GetImageData().GetDimensions()[2], 992)

    self.delayDisplay('Test passed')
