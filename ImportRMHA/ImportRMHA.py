# vim: set tabstop=2 softtabstop=2 shiftwidth=2 expandtab

import os
import unittest
import logging
import vtk, qt, ctk, slicer
import slicer
import numpy as np

from slicer.ScriptedLoadableModule import *

import SimpleITK as sitk
import sitkUtils
from multiprocessing import cpu_count

from pathlib import Path

#
# ImportRMHA
#

"""
{'ObjectType': 'ROI',
 'NDims': 3,
 'BinaryData': True,
 'BinaryDataByteOrderMSB': False,
 'CompressedData': 'RLE',
 'TransformMatrix': [-1, 0, 0, 0, -1, 0, 0, 0, -1],
 'Offset': [0, 0, 0],
 'CenterOfRotation': [0, 0, 0],
 'AnatomicalOrientation': 'LPS',
 'ElementSpacing': [0.020446, 0.020446, 0.020446],
 'DimSize': [490, 490, 587],
 'ElementType': numpy.uint8,
 'ROI[1]': 'Vis:red:1:0:127:255',
 'ROI[2]': 'Implant:yellow:0:0:127:255',
 'ROI[3]': 'Upper Bone:green:0:0:127:255',
 'ROI[4]': 'Lower Bone:blue:0:0:127:255',
 'PatientsName': '',
 'PatientID': '',
 'StudyDescription': '',
 'StudyDate': '',
 'StudyTime': '',
 'SeriesDescription': '',
 'SeriesDate': '',
 'SeriesTime': '',
 'ReferenceUID': '',
 'ElementDataFile': 'LOCAL'}
"""
def string_to_nums(s):
    if ' ' in s:
        ret = [float(v) if '.' in v else int(v) for v in s.split(' ')]
    else:
        ret = float(s) if '.' in s else int(s)
    return ret

def is_alphanumspacedot(s):
    if s == '':
        return False
    
    ret = True
    for c in s:
        if not (c.isdigit() or c==' ' or c=='.' or c=='-'):
            ret = False
            break
    return ret

def read_rmha(fn):
    chunk = ''
    dic = {}
    f = open(fn, "rb")
    while 1:
        byte = f.read(1)
        if byte == b'\0':
            #Go back one position and read the rest...
            f.seek(-1, 1)
            rle = f.read()
            f.close()
            break
        elif byte == b'\n':
            key, val = chunk.split(' = ', 1)
            dic[key] = val
            chunk = ''
        else:
            #print(ord(byte), byte.decode('ascii', 'ignore'))
            chunk += byte.decode('ascii', 'ignore')
            
    print(len(rle))
    #Now convert the dic
    for key,val in dic.items():
        if val == 'True':
            val = True
        elif val == 'False':
            val = False
        elif val == 'MET_UCHAR':
            val = np.uint8
        elif is_alphanumspacedot(val):
            val = string_to_nums(val)
        
        dic[key] = val

    img = np.zeros(np.product(dic['DimSize']), dtype=dic['ElementType'])
    
    cnt = 0
    for i in range(0, len(rle), 2):
        img[cnt:cnt + rle[i+1]] = rle[i]
        cnt += rle[i+1]
        
    img = img.reshape(dic['DimSize'][::-1])
    return img, dic


class ImportRMHA(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ImportRMHA"
    self.parent.categories = ["Informatics"]
    self.parent.dependencies = []
    self.parent.contributors = ["Egor Zindy (Université Libre Bruxelles), Andras Lasso (PerkLab)"]
    self.parent.helpText = """Load RMHA image volumes"""
    self.parent.acknowledgementText = """This file borrows heavily from ImportOCT, which was originally developed by Andras Lasso, PerkLab."""
    # don't show this module - it is only for registering a reader
    parent.hidden = True 

#
# Reader plugin
# (identified by its special name <moduleName>FileReader)
#

class ImportRMHAFileReader(object):

  def __init__(self, parent):
    self.parent = parent

  def description(self):
    return 'RMHA image'

  def fileType(self):
    return 'RMHAImageFile'

  def extensions(self):
    # TODO: we could expose all file formats that oct-converter Python package supports,
    # but we would need sample data sets to test with
    return ['RMHA image file (*.rmha)']

  def checkRequiredPythonPackages(self):
    return True

  def canLoadFile(self, filePath):
    if not self.checkRequiredPythonPackages():
      return False
    return True

  def load(self, properties):
    if not self.checkRequiredPythonPackages():
      return False

    try:
      filePath = properties['fileName']
      img_array, dic = read_rmha(filePath)
      n_zs, n_ys, n_xs = img_array.shape 

      #From here we should be able to get the data...
      base_name = dic['ObjectType']
      base_name = slicer.mrmlScene.GenerateUniqueName(base_name)

      volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', base_name)
      #'ElementSpacing': [0.020446, 0.020446, 0.020446],
      #'DimSize': [490, 490, 587],
      XScale, YScale, ZScale = dic['ElementSpacing'] # Spacing already in mm
      volumeNode.SetSpacing(XScale, YScale, ZScale)

      # 'TransformMatrix': [-1, 0, 0, 0, -1, 0, 0, 0, -1],
      volumeNode.SetOrigin(*dic['Offset'])
      # 'Offset': [0, 0, 0],

      # 'CenterOfRotation': [0, 0, 0],

      slicer.util.updateVolumeFromArray(volumeNode, img_array)
      slicer.util.setSliceViewerLayers(volumeNode, fit=True)

      scalarRange = [np.min(img_array), np.max(img_array)]

      black=[0.,0.,0.]  
      white=[1.,1.,1.]

      colorTransfer= self.createColorTransfer(black, white, scalarRange)
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
# ImportRMHATest
#

class ImportRMHATest(ScriptedLoadableModuleTest):

  def setUp(self):
    slicer.mrmlScene.Clear()

  def runTest(self):
    self.setUp()
    self.test_ImportRMHA1()

  def test_ImportRMHA1(self):

    self.delayDisplay("Loading test image as segmentation")

    import SampleData
    testFdaFilePath = SampleData.downloadFromURL(
      fileNames='FIXME.czi',
      uris='https://github.com/PerkLab/SlicerSandbox/releases/download/TestingData/FIXME.czi',
      checksums='SHA256:5536006a2bb4d117f5804e49d393ecc1cb7e98c3e5f7924b7b83b8f0e0567e2c')[0]
    volumeNode = slicer.util.loadNodeFromFile(testFdaFilePath, 'RMHAImageFile')
    self.assertIsNotNone(volumeNode)

    # FIXME -- Find an image, check these things...
    self.delayDisplay('Checking loaded image')
    self.assertEqual(volumeNode.GetImageData().GetDimensions()[0], 9)
    self.assertEqual(volumeNode.GetImageData().GetDimensions()[1], 320)
    self.assertEqual(volumeNode.GetImageData().GetDimensions()[2], 992)

    self.delayDisplay('Test passed')
