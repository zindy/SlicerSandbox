import contextlib
import logging
import os
import sys
import types
import uuid
from pathlib import Path

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *

#
# ImportMicroscopy
#
# Based on ImportCZI. Instead of the czitools_lite fork, image files are opened with
# napari-virtual-reader (https://github.com/sebi06/napari-virtual-reader), which itself
# relies on refkv (https://github.com/d-v-b/refkv) to expose a file as a lazy, virtual
# OME-Zarr image without converting or copying it.
#

# File formats offered to Slicer's "Add Data" dialog.
#
# Deliberately NOT listed, even though napari-virtual-reader accepts them: *.dcm, *.nii,
# *.nii.gz, *.nrrd, *.nhdr (Slicer has its own, better readers), and plain *.tif / *.tiff
# (see INCLUDE_GENERIC_FORMATS). Claiming those would make this module compete with
# Slicer's native readers on every load. Whether a given format actually opens is decided by
# refkv's parser, not by this list.
FORMATS = [
  ("Zeiss CZI image", (".czi",)),
  ("Nikon ND2 image", (".nd2",)),
  ("Leica LIF image", (".lif",)),
  ("Imaris IMS image", (".ims",)),
  ("Olympus VSI image", (".vsi",)),
  ("Whole-slide image (SVS, NDPI, QPTIFF)", (".svs", ".ndpi", ".qptiff")),
  ("OME-TIFF image", (".ome.tif", ".ome.tiff")),
  ("refkv virtual archive", (".vzip",)),
]

# Set to True to also offer plain TIFF files (e.g. ImageJ hyperstacks). Slicer's own TIFF
# reader will then compete for them; ours reports a lower confidence, so it is only picked
# when chosen explicitly in the Add Data dialog's "Show options" reader drop-down.
INCLUDE_GENERIC_FORMATS = False
GENERIC_FORMATS = [("TIFF microscopy image", (".tif", ".tiff"))]

# Never turn a single volume into more than this many megabytes. For files with a stored
# pyramid the coarsest-needed level is picked automatically; see "maxSizeMB" below.
DEFAULT_MAX_SIZE_MB = 2048

# Starting points for the volume rendering opacity ramp (adjustable afterwards in Slicer):
# opacity is 0 below  median + LOW_SIGMA * (robust std of the background),  and reaches 1 at the
# HIGH_QUANTILE percentile of the intensities.
LOW_SIGMA = 3.0
HIGH_QUANTILE = 0.999

# pip requirement for the reader itself. Pin a version here if you need reproducibility.
PLUGIN_REQUIREMENT = "napari-virtual-reader"

# OME unit name -> millimeters (Slicer's internal length unit)
UNIT_TO_MM = {
  "angstrom": 1e-7, "picometer": 1e-9, "nanometer": 1e-6, "nm": 1e-6,
  "micrometer": 1e-3, "micron": 1e-3, "um": 1e-3, "µm": 1e-3, "μm": 1e-3,
  "millimeter": 1.0, "mm": 1.0, "centimeter": 10.0, "cm": 10.0,
  "meter": 1000.0, "m": 1000.0, "kilometer": 1e6, "km": 1e6,
}


def _allFormats():
  return FORMATS + (GENERIC_FORMATS if INCLUDE_GENERIC_FORMATS else [])


def _supportedSuffixes():
  return tuple(s for _, suffixes in _allFormats() for s in suffixes)


#
# ImportMicroscopy
#

class ImportMicroscopy(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ImportMicroscopy"
    self.parent.categories = ["Informatics"]
    self.parent.dependencies = []
    self.parent.contributors = ["Egor Zindy (Université Libre de Bruxelles), Andras Lasso (PerkLab)"]
    self.parent.helpText = """Load microscopy image volumes (CZI, ND2, LIF, IMS, VSI, SVS, NDPI, QPTIFF, OME-TIFF, ...)
using napari-virtual-reader / refkv. Each channel and timepoint becomes a volume node.
<br>Optional properties for slicer.util.loadNodeFromFile(path, 'MicroscopyImageFile', properties):
<b>scene</b> (index or 'all'; if omitted a chooser is shown for multi-image files), <b>timepoint</b> (index, default: all),
<b>maxSizeMB</b> (largest volume to create, default %d; a coarser pyramid level is used if needed),
<b>resetViews</b> (fit slice views, re-centre the 3D camera and set its orthographic scale to the loaded image, default True),
<b>showDisplayAdjustment</b> (open the channel display dialog after loading, default True),
<b>style3DView</b> (orthographic projection, black background and scale bar in the 3D views, default True; implies resetting the 3D camera)."""  % DEFAULT_MAX_SIZE_MB
    self.parent.acknowledgementText = """This file is based on ImportCZI and ImportOCT, originally developed by Andras Lasso, PerkLab.
Image access is provided by napari-virtual-reader (Sebastian Rhode, BSD-3-Clause) and refkv (Davis Bennett)."""
    # Visible so that the Display Adjustment dialog can be reopened from the module panel
    parent.hidden = False


class ImportMicroscopyWidget(ScriptedLoadableModuleWidget):
  """Module panel: reopen the Display Adjustment dialog for images loaded by this module."""

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.layout.addWidget(qt.QLabel(
      "Images are loaded through File > Add Data (or Add Data dialog drag and drop).\n"
      "Reopen the channel display settings of a loaded image here."))
    self.groupCombo = qt.QComboBox()
    self.layout.addWidget(self.groupCombo)
    self.openButton = qt.QPushButton("Display adjustment...")
    self.openButton.clicked.connect(lambda checked=False: self.onOpen())
    self.layout.addWidget(self.openButton)
    self.layout.addStretch(1)
    self.refresh()

  def enter(self):
    self.refresh()

  def refresh(self):
    self.groupCombo.clear()
    groups = findImageGroups()
    for groupId, nodes in groups.items():
      self.groupCombo.addItem(nodes[0].GetAttribute(ATTR + "groupLabel") or groupId, groupId)
    self.openButton.setEnabled(len(groups) > 0)

  def onOpen(self):
    showDisplayAdjustment(self.groupCombo.itemData(self.groupCombo.currentIndex))


#
# napari stand-in
#

@contextlib.contextmanager
def _napariShim():
  """napari-virtual-reader's read_images() imports exactly two things from napari (a Colormap
  class and show_warning), both lazily, inside a function. Installing napari (and a Qt binding)
  into Slicer would be heavy and conflict with Slicer's own Qt, so if napari is not available
  we provide minimal stand-ins, only for the duration of the call, and remove them afterwards
  so that other Slicer extensions probing for napari are not misled."""
  import importlib.util
  try:
    haveNapari = importlib.util.find_spec("napari") is not None
  except (ImportError, ValueError):
    haveNapari = False
  if haveNapari:
    yield
    return

  class Colormap:
    def __init__(self, colors, name="custom", **kwargs):
      self.colors = colors
      self.name = name

  def show_warning(message):
    logging.warning(str(message))

  shims = {}
  for name in ("napari", "napari.utils", "napari.utils.colormaps", "napari.utils.notifications"):
    shims[name] = types.ModuleType(name)
  shims["napari.utils.colormaps"].Colormap = Colormap
  shims["napari.utils.notifications"].show_warning = show_warning

  added = [name for name in shims if name not in sys.modules]
  for name in added:
    sys.modules[name] = shims[name]
  try:
    yield
  finally:
    for name in added:
      if sys.modules.get(name) is shims[name]:
        del sys.modules[name]


#
# Layer handling (independent of Slicer, works on napari layer-data tuples)
#

def _hexToRgb(color):
  """'RRGGBB', '#RRGGBB' or '#AARRGGBB' -> (r, g, b) in 0-1, or None."""
  if not isinstance(color, str):
    return None
  c = color.lstrip("#")
  if len(c) == 8:
    c = c[2:]
  if len(c) != 6:
    return None
  try:
    return [int(c[i:i + 2], 16) / 255. for i in (0, 2, 4)]
  except ValueError:
    return None


def collectScenes(layers):
  """Regroup napari layer-data tuples as {scene_key: [channel entries]}.

  read_images() either stacks compatible scenes on a leading S axis (one layer per channel),
  or keeps one layer per (scene, channel). Both are flattened here into per-scene, per-channel
  entries so that the rest of the code does not care. Each entry holds:
    levels     list of lazy arrays, finest first, axes = labels (+ trailing RGB(A) axis if rgb)
    labels     lower-case axis names, e.g. ['t', 'z', 'y', 'x']
    scale / translate   per label, in the units given in `units`
    units      per label (OME unit name or None)
  """
  scenes = {}
  for data, kw, _ in layers:
    meta = kw["metadata"]
    levels = list(data) if kw.get("multiscale") else [data]
    labels = [str(a).lower() for a in kw["axis_labels"]]
    ome = meta.get("ome", {})
    multiscales = ome.get("multiscales", [{}])
    omeAxes = multiscales[min(meta.get("multiscale_index", 0), len(multiscales) - 1)].get("axes", [])
    unitOf = {}
    for ax in omeAxes:
      if isinstance(ax, dict):
        unitOf[ax["name"].lower()] = ax.get("unit")
    channels = ome.get("omero", {}).get("channels", [])
    chIndex = meta.get("channel", 0)
    seriesNames = [m.get("name") for m in multiscales]
    seriesName = seriesNames[min(meta.get("multiscale_index", 0), len(seriesNames) - 1)]
    color = channels[chIndex].get("color") if chIndex < len(channels) else None

    def makeEntry(lv, lab, scale, translate, name=seriesName):
      return {
        "levels": lv, "labels": lab, "series_name": name,
        "scale": list(scale), "translate": list(translate),
        "units": [unitOf.get(a) for a in lab],
        "rgb": bool(kw.get("rgb")),
        "limits": kw.get("contrast_limits"),
        "channel": chIndex,
        "label": meta.get("channel_label", f"Channel {chIndex}"),
        "color": _hexToRgb(color),
      }

    if labels and labels[0] == "s":
      for s, scene in enumerate(meta["scenes"]):
        key = (scene["series"], meta.get("multiscale_index", 0))
        sceneMultiscales = scene.get("ome", {}).get("multiscales", [{}])
        entry = makeEntry([lv[s] for lv in levels], labels[1:], scene["scale"], scene["translate"],
                          sceneMultiscales[0].get("name"))
        scenes.setdefault(key, []).append(entry)
    else:
      key = (meta.get("series", ""), meta.get("multiscale_index", 0))
      entry = makeEntry(levels, labels, kw["scale"], kw["translate"])
      scenes.setdefault(key, []).append(entry)

  for entries in scenes.values():
    entries.sort(key=lambda e: e["channel"])
  return scenes


def describeScene(key, entries):
  """One-line description for the scene chooser, e.g. 'Series003 - 2 ch, X512 Y512 Z20 T3'."""
  first = entries[0]
  level = first["levels"][0]
  dims = " ".join(f"{a.upper()}{level.shape[i]}" for i, a in reversed(list(enumerate(first["labels"]))))
  name = first.get("series_name") or key[0] or "image"
  return f"{name} - {len(entries)} ch, {dims}"


def timepointCount(entry):
  labels = entry["labels"]
  return entry["levels"][0].shape[labels.index("t")] if "t" in labels else 1


def planEntry(entry, maxBytes):
  """Pick the pyramid level to load. Returns (levelIndex, nbytesOfOneTimepoint, fits)."""
  labels = entry["labels"]
  best = None
  for i, level in enumerate(entry["levels"]):
    n = 1
    for axis, size in enumerate(level.shape):
      if axis < len(labels) and labels[axis] == "t":
        continue
      n *= size
    nbytes = n * level.dtype.itemsize
    best = (i, nbytes)
    if nbytes <= maxBytes:
      return i, nbytes, True
  return best[0], best[1], False


def geometry(entry, levelIndex):
  """Spacing and origin in mm (x, y, z) for the chosen level."""
  labels = entry["labels"]
  level0 = entry["levels"][0]
  level = entry["levels"][levelIndex]
  spacing, origin, factors = {}, {}, {}
  for ax in ("x", "y", "z"):
    if ax not in labels:
      continue
    i = labels.index(ax)
    unit = entry["units"][i]
    factor = UNIT_TO_MM.get(str(unit).lower(), 1.0) if unit else 1.0
    ratio = level0.shape[i] / level.shape[i]   # >1 for coarser pyramid levels
    spacing[ax] = entry["scale"][i] * ratio * factor
    origin[ax] = entry["translate"][i] * factor
    factors[ax] = factor
  if "z" not in spacing:
    # a single 2D plane: give it the in-plane thickness
    spacing["z"] = spacing.get("x", 1.0)
    origin["z"] = 0.0
  return ([spacing.get("x", 1.0), spacing.get("y", 1.0), spacing["z"]],
          [origin.get("x", 0.0), origin.get("y", 0.0), origin["z"]])


def readVoxels(entry, levelIndex, t=0):
  """Compute one timepoint of one channel -> numpy (z, y, x) or (z, y, x, components)."""
  import numpy as np

  labels = entry["labels"]
  level = entry["levels"][levelIndex]
  index, kept = [], []
  for axis, name in enumerate(labels):
    if name == "t":
      index.append(t)
    elif name in ("z", "y", "x"):
      index.append(slice(None))
      kept.append(name)
    else:
      logging.warning(f"Unexpected axis '{name}': using index 0")
      index.append(0)
  if entry["rgb"]:
    index.append(slice(None))
  arr = np.asarray(level[tuple(index)].compute())

  order = [kept.index(a) for a in ("z", "y", "x") if a in kept]
  if entry["rgb"]:
    order.append(arr.ndim - 1)
  arr = np.transpose(arr, order)
  if "z" not in kept:
    arr = arr[np.newaxis]
  return arr


def normalizeDtype(arr):
  import numpy as np
  if arr.dtype == np.bool_:
    return arr.astype(np.uint8)
  if arr.dtype == np.float16:
    return arr.astype(np.float32)
  if arr.dtype in (np.int64, np.uint64):
    return arr.astype(np.float64)
  if np.issubdtype(arr.dtype, np.complexfloating):
    raise ValueError("Complex-valued images are not supported")
  return arr


def toRgb8(arr, limits):
  """Packed colour -> uint8 RGB (alpha dropped), scaled with the display window if needed."""
  import numpy as np
  arr = arr[..., :3]
  if arr.dtype == np.uint8:
    return np.ascontiguousarray(arr)
  low, high = (limits if limits else (float(arr.min()), float(arr.max())))
  if high <= low:
    high = low + 1
  scaled = np.clip((arr.astype(np.float64) - low) / (high - low), 0, 1) * 255
  return scaled.astype(np.uint8)


#
# Volume rendering and display adjustment
#
# Fluorescence-style rendering: every channel has ONE solid colour, and contrast is controlled
# by the scalar opacity curve alone: opacity is 0 up to `min`, rises to 1 at `max` (shaped by
# `gamma`), and stays 1 above. The same settings are used by the loader (as starting values)
# and by the Display Adjustment dialog, and are stored as a JSON node attribute so that they
# survive saving and reloading the scene.
#

ATTR = "ImportMicroscopy."
GAMMA_STEPS = 16


def estimateOpacityRamp(values, vmin, vmax):
  """Return (low, high). high is the HIGH_QUANTILE percentile; low is the background level
  (median) plus LOW_SIGMA robust standard deviations, but never less than 5% of the way to
  high, so that images with a flat zero background still get a sensible threshold."""
  import numpy as np
  high = float(np.quantile(values, HIGH_QUANTILE))
  background = float(np.median(values))
  sigma = 1.4826 * float(np.median(np.abs(values - background)))
  low = max(background + LOW_SIGMA * sigma, vmin + 0.05 * (high - vmin))
  if not low < high:
    # almost no signal above background: fall back to a ramp across the upper half of the range
    high = vmax
    low = vmin + 0.5 * (vmax - vmin)
  return low, high


def sampleVoxels(array):
  """Coarse subsample (every 4th voxel along each axis) as a flat float64 vector."""
  import numpy as np
  step = tuple(4 if d > 4 else 1 for d in array.shape)
  return array[tuple(slice(None, None, s) for s in step)].astype(np.float64).ravel()


def opacityPoints(rangeMin, rangeMax, low, high, gamma, steps=GAMMA_STEPS):
  """Scalar opacity curve as [(x, opacity)]: 0 up to low, ramp to 1 at high, 1 above.
  Between low and high opacity = t ** (1 / gamma), with t going from 0 to 1, so gamma > 1
  brightens the mid-range and gamma < 1 darkens it. gamma = 1 is a straight line."""
  points = [(rangeMin, 0.0)]
  if low > rangeMin:
    points.append((low, 0.0))
  if abs(gamma - 1.0) > 1e-6:
    for i in range(1, steps):
      t = i / steps
      points.append((low + t * (high - low), t ** (1.0 / gamma)))
  points.append((high, 1.0))
  if rangeMax > high:
    points.append((rangeMax, 1.0))
  return points


def histogramGeometry(counts, vmin, vmax, low, high, gamma, width, height, stripHeight):
  """Pixel geometry for the dialog's histogram, independent of Qt (so it can be tested).
  Returns bars [(x0, x1, barHeight)], curve [(x, y)], xLow, xHigh, strip [alpha per pixel column]."""
  import numpy as np
  counts = np.asarray(counts, dtype=np.float64)
  n = len(counts)
  span = (vmax - vmin) or 1.0
  plotH = height - stripHeight
  heights = np.log1p(counts)                       # log scale, like Imaris
  if heights.max() > 0:
    heights = heights / heights.max()
  bars = [(int(round(i * width / n)), int(round((i + 1) * width / n)), int(round(h * (plotH - 1))))
          for i, h in enumerate(heights)]

  def xOf(v):
    return min(max(int(round((v - vmin) / span * (width - 1))), 0), width - 1)

  xLow, xHigh = xOf(low), xOf(high)
  bottom, top = plotH - 1, 0
  curve = [(0, bottom), (xLow, bottom)]
  strip = []
  for x in range(width):
    if x <= xLow:
      alpha = 0.0
    elif x >= xHigh:
      alpha = 1.0
    else:
      alpha = ((x - xLow) / (xHigh - xLow)) ** (1.0 / gamma)
    strip.append(alpha)
    if xLow < x < xHigh:
      curve.append((x, int(round(bottom - alpha * (bottom - top)))))
  curve += [(xHigh, top), (width - 1, top)]
  return {"bars": bars, "curve": curve, "xLow": xLow, "xHigh": xHigh, "strip": strip}


def loadSettings(node):
  import json
  text = node.GetAttribute(ATTR + "display")
  if not text:
    return None
  try:
    return json.loads(text)
  except ValueError:
    return None


def saveSettings(node, settings):
  import json
  node.SetAttribute(ATTR + "display", json.dumps(settings))


def createVolumePropertyNode():
  volumePropertyNode = slicer.vtkMRMLVolumePropertyNode()
  slicer.mrmlScene.RegisterNodeClass(volumePropertyNode)
  vtkVolumeProperty = volumePropertyNode.GetVolumeProperty()
  vtkVolumeProperty.SetInterpolationTypeToNearest()
  vtkVolumeProperty.ShadeOn()
  vtkVolumeProperty.SetAmbient(0.30)
  vtkVolumeProperty.SetDiffuse(0.60)
  vtkVolumeProperty.SetSpecular(0.50)
  vtkVolumeProperty.SetSpecularPower(40)
  slicer.mrmlScene.AddNode(volumePropertyNode)
  return volumePropertyNode


def volumeRenderingNodes(volumeNode):
  """The volume rendering display node and volume property node of a volume (created if needed)."""
  logic = slicer.modules.volumerendering.logic()
  displayNode = logic.GetFirstVolumeRenderingDisplayNode(volumeNode)
  if displayNode is None:
    displayNode = logic.CreateDefaultVolumeRenderingNodes(volumeNode)
    displayNode.SetAndObserveVolumePropertyNodeID(createVolumePropertyNode().GetID())
  return displayNode, displayNode.GetVolumePropertyNode()


def applyRendering(volumeNode, settings):
  """Push {color, min, max, gamma, visible} into the volume rendering of one volume node."""
  displayNode, propertyNode = volumeRenderingNodes(volumeNode)
  scalarRange = volumeNode.GetImageData().GetScalarRange()
  low, high = float(settings["min"]), float(settings["max"])
  if high <= low:
    high = low + max(abs(low) * 1e-6, 1e-6)
  rangeMin, rangeMax = min(scalarRange[0], low), max(scalarRange[1], high)
  color = settings["color"]

  colorTransfer = vtk.vtkColorTransferFunction()        # one solid colour at every intensity
  colorTransfer.AddRGBPoint(rangeMin, color[0], color[1], color[2])
  colorTransfer.AddRGBPoint(rangeMax, color[0], color[1], color[2])
  opacity = vtk.vtkPiecewiseFunction()
  for x, alpha in opacityPoints(rangeMin, rangeMax, low, high, float(settings.get("gamma", 1.0))):
    opacity.AddPoint(x, alpha)
  propertyNode.SetColor(colorTransfer)
  propertyNode.SetScalarOpacity(opacity)
  displayNode.SetVisibility(bool(settings.get("visible", True)))


class ChannelDisplay(object):
  """One channel of one image: its volume nodes (one per timepoint) share one set of settings."""

  def __init__(self, label, nodes):
    self.label = label
    self.nodes = nodes
    stored = loadSettings(nodes[0])
    self.settings = stored or {"color": [1., 1., 1.], "min": 0., "max": 1., "gamma": 1., "visible": True}
    ranges = [n.GetImageData().GetScalarRange() for n in nodes]
    self.vmin = min(r[0] for r in ranges)
    self.vmax = max(r[1] for r in ranges)
    if self.vmax <= self.vmin:
      self.vmax = self.vmin + 1.
    self._sample = None

  def sample(self):
    if self._sample is None:
      self._sample = sampleVoxels(slicer.util.arrayFromVolume(self.nodes[0]))
    return self._sample

  def histogram(self, bins=256):
    import numpy as np
    return np.histogram(self.sample(), bins=bins, range=(self.vmin, self.vmax))[0]

  def auto(self):
    self.settings["min"], self.settings["max"] = estimateOpacityRamp(self.sample(), self.vmin, self.vmax)
    self.settings["gamma"] = 1.0

  def reset(self):
    self.settings["min"], self.settings["max"], self.settings["gamma"] = self.vmin, self.vmax, 1.0

  def apply(self):
    for node in self.nodes:
      try:
        applyRendering(node, self.settings)
        saveSettings(node, self.settings)
      except Exception as e:
        logging.warning(f"Could not update volume rendering of {node.GetName()}: {e}")


def findImageGroups():
  """{group id: [volume nodes]} for all images loaded by this module that can be adjusted."""
  groups = {}
  for node in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode'):
    groupId = node.GetAttribute(ATTR + "group")
    if groupId:
      groups.setdefault(groupId, []).append(node)
  return groups


def channelsOfGroup(nodes):
  byChannel = {}
  for node in nodes:
    byChannel.setdefault(int(node.GetAttribute(ATTR + "channel") or 0), []).append(node)
  return [ChannelDisplay(group[0].GetAttribute(ATTR + "label") or f"Channel {index}", group)
          for index, group in sorted(byChannel.items())]


_openDialogs = {}


def showDisplayAdjustment(groupId=None):
  """Open (or raise) the Display Adjustment dialog for an image loaded by this module."""
  groups = findImageGroups()
  if not groups:
    slicer.util.infoDisplay("No microscopy volumes with adjustable display were found in the scene.")
    return None
  if groupId not in groups:
    groupId = list(groups)[-1]
  existing = _openDialogs.get(groupId)
  if existing is not None:
    existing.dialog.show()
    existing.dialog.raise_()
    existing.dialog.activateWindow()
    return existing
  nodes = groups[groupId]
  title = "Display Adjustment - " + (nodes[0].GetAttribute(ATTR + "groupLabel") or "")
  dialog = DisplayAdjustmentDialog(channelsOfGroup(nodes), title.strip(" -"), slicer.util.mainWindow())
  _openDialogs[groupId] = dialog
  dialog.dialog.finished.connect(lambda *args, g=groupId: _openDialogs.pop(g, None))
  dialog.dialog.show()
  return dialog


class DisplayAdjustmentDialog(object):
  """Imaris-style channel display dialog: per-channel visibility and solid colour, with the
  minimum, maximum and gamma of the opacity curve over a histogram of the channel."""

  WIDTH = 380
  HIST_HEIGHT = 120
  STRIP_HEIGHT = 14

  def __init__(self, channels, title="Display Adjustment", parent=None):
    self.channels = channels
    self.current = 0
    self._updating = False
    self.dialog = qt.QDialog(parent)
    self.dialog.setWindowTitle(title)
    layout = qt.QVBoxLayout(self.dialog)

    # One row per channel: visibility check box + a button in the channel's colour (selects it)
    self.channelButtons, self.channelChecks = [], []
    self.buttonGroup = qt.QButtonGroup(self.dialog)
    for i, channel in enumerate(channels):
      row = qt.QHBoxLayout()
      check = qt.QCheckBox()
      check.setChecked(bool(channel.settings.get("visible", True)))
      check.setToolTip("Show or hide this channel in the 3D view")
      check.toggled.connect(lambda state, i=i: self._onVisible(i, state))
      button = qt.QPushButton(channel.label)
      button.setCheckable(True)
      button.setMinimumHeight(30)
      button.clicked.connect(lambda checked=False, i=i: self._select(i))
      self.buttonGroup.addButton(button)
      row.addWidget(check)
      row.addWidget(button, 1)
      layout.addLayout(row)
      self.channelChecks.append(check)
      self.channelButtons.append(button)

    self.histogramLabel = qt.QLabel()
    self.histogramLabel.setFixedSize(self.WIDTH, self.HIST_HEIGHT)
    layout.addWidget(self.histogramLabel)

    self.showAllCheck = qt.QCheckBox("Show/Hide all channels")
    self.showAllCheck.toggled.connect(self._onShowAll)
    layout.addWidget(self.showAllCheck)
    autoButton = qt.QPushButton("Auto adjust all channels")
    autoButton.setToolTip("Set min and max from the image statistics: just above the background noise, up to the 99.9th percentile")
    autoButton.clicked.connect(lambda checked=False: self._autoAll())
    resetButton = qt.QPushButton("Reset all channels")
    resetButton.setToolTip("Min and max at the full intensity range of each channel, gamma 1")
    resetButton.clicked.connect(lambda checked=False: self._resetAll())
    layout.addWidget(autoButton)
    layout.addWidget(resetButton)

    header = qt.QHBoxLayout()
    self.settingsLabel = qt.QLabel()
    colorButton = qt.QPushButton("Colour...")
    colorButton.clicked.connect(lambda checked=False: self._onColor())
    header.addWidget(self.settingsLabel, 1)
    header.addWidget(colorButton)
    layout.addLayout(header)

    grid = qt.QGridLayout()
    self.spins, self.sliders = {}, {}
    for r, (key, name) in enumerate((("min", "Min:"), ("max", "Max:"), ("gamma", "Gamma:"))):
      spin = qt.QDoubleSpinBox()
      slider = qt.QSlider(qt.Qt.Horizontal)
      if key == "gamma":
        slider.setRange(10, 500)
      else:
        slider.setRange(0, 1000)
      spin.valueChanged.connect(lambda value, key=key: self._onSpin(key, value))
      slider.valueChanged.connect(lambda value, key=key: self._onSlider(key, value))
      grid.addWidget(qt.QLabel(name), r, 0)
      grid.addWidget(spin, r, 1)
      grid.addWidget(slider, r, 2)
      self.spins[key], self.sliders[key] = spin, slider
    self.spins["gamma"].setToolTip("Shapes the opacity between min and max. Above 1 brightens the mid-range, below 1 darkens it.")
    layout.addLayout(grid)

    self.channelButtons[0].setChecked(True)
    self._select(0)

  # ---- helpers

  @property
  def channel(self):
    return self.channels[self.current]

  def _span(self):
    return self.channel.vmax - self.channel.vmin

  def _minGap(self):
    return self._span() / 1000.

  def _select(self, index):
    self.current = index
    channel = self.channel
    span = self._span()
    decimals = 4 if span < 10 else 2
    self._updating = True
    for key in ("min", "max"):
      spin = self.spins[key]
      spin.setDecimals(decimals)
      spin.setRange(channel.vmin, channel.vmax)
      spin.setSingleStep(span / 200.)
    self.spins["gamma"].setDecimals(2)
    self.spins["gamma"].setRange(0.1, 5.0)
    self.spins["gamma"].setSingleStep(0.05)
    self._updating = False
    self._refresh()

  def _refresh(self):
    """Update every widget from the current channel's settings."""
    channel, s = self.channel, self.channel.settings
    self._updating = True
    self.settingsLabel.setText(f"Settings - {channel.label}")
    for key in ("min", "max", "gamma"):
      self.spins[key].setValue(s[key])
    span = self._span()
    for key in ("min", "max"):
      self.sliders[key].setValue(int(round((s[key] - channel.vmin) / span * 1000)))
    self.sliders["gamma"].setValue(int(round(s["gamma"] * 100)))
    for i, ch in enumerate(self.channels):
      self.channelChecks[i].setChecked(bool(ch.settings["visible"]))
      self._styleRow(i)
    self.showAllCheck.setChecked(all(ch.settings["visible"] for ch in self.channels))
    self._updating = False
    self._paintHistogram()

  def _styleRow(self, i):
    r, g, b = [int(round(c * 255)) for c in self.channels[i].settings["color"]]
    self.channelButtons[i].setStyleSheet(
      "QPushButton { text-align: left; padding-left: 8px; color: white; border: 2px solid transparent;"
      f" background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgb(20,20,20), stop:1 rgb({r},{g},{b})); }}"
      "QPushButton:checked { border: 2px solid #e0c040; }")

  def _commit(self):
    self.channel.apply()
    self._refresh()

  # ---- widget callbacks

  def _onVisible(self, index, state):
    if self._updating:
      return
    self.channels[index].settings["visible"] = bool(state)
    self.channels[index].apply()
    self._refresh()

  def _onShowAll(self, state):
    if self._updating:
      return
    for ch in self.channels:
      ch.settings["visible"] = bool(state)
      ch.apply()
    self._refresh()

  def _onSpin(self, key, value):
    if self._updating:
      return
    s, gap = self.channel.settings, self._minGap()
    if key == "min":
      value = min(value, s["max"] - gap)
    elif key == "max":
      value = max(value, s["min"] + gap)
    s[key] = value
    self._commit()

  def _onSlider(self, key, position):
    if self._updating:
      return
    if key == "gamma":
      value = position / 100.
    else:
      value = self.channel.vmin + position / 1000. * self._span()
    self._onSpin(key, value)

  def _onColor(self):
    s = self.channel.settings
    initial = qt.QColor(*[int(round(c * 255)) for c in s["color"]])
    chosen = qt.QColorDialog.getColor(initial, self.dialog, "Channel colour")
    if chosen.isValid():
      s["color"] = [chosen.red() / 255., chosen.green() / 255., chosen.blue() / 255.]
      self._commit()

  def _autoAll(self):
    for ch in self.channels:
      ch.auto()
      ch.apply()
    self._refresh()

  def _resetAll(self):
    for ch in self.channels:
      ch.reset()
      ch.apply()
    self._refresh()

  # ---- histogram

  def _paintHistogram(self):
    channel, s = self.channel, self.channel.settings
    W, H, S = self.WIDTH, self.HIST_HEIGHT, self.STRIP_HEIGHT
    plotH = H - S
    geometry = histogramGeometry(channel.histogram(), channel.vmin, channel.vmax,
                                 s["min"], s["max"], s["gamma"], W, H, S)
    color = [int(round(c * 255)) for c in s["color"]]
    background = (28, 28, 28)

    pixmap = qt.QPixmap(W, H)
    pixmap.fill(qt.QColor(*background))
    painter = qt.QPainter(pixmap)
    grey = qt.QColor(125, 125, 125)
    for x0, x1, h in geometry["bars"]:
      if h > 0:
        painter.fillRect(x0, plotH - h, max(1, x1 - x0), h, grey)
    painter.setPen(qt.QColor(*color))                     # the opacity curve, in the channel colour
    curve = geometry["curve"]
    for (xa, ya), (xb, yb) in zip(curve[:-1], curve[1:]):
      painter.drawLine(xa, ya, xb, yb)
    painter.setPen(qt.QColor(255, 255, 255))              # min and max markers
    for x in (geometry["xLow"], geometry["xHigh"]):
      painter.drawLine(x, 0, x, plotH - 1)
    for x, alpha in enumerate(geometry["strip"]):         # colour strip: what each intensity looks like
      mixed = [int(round(background[k] * (1 - alpha) + color[k] * alpha)) for k in range(3)]
      painter.fillRect(x, plotH, 1, S, qt.QColor(*mixed))
    painter.setPen(qt.QColor(255, 255, 255))
    painter.drawText(3, H - 2, f"{channel.vmin:g}")
    right = f"{channel.vmax:g}"
    painter.drawText(W - 3 - painter.fontMetrics().boundingRect(right).width(), H - 2, right)
    painter.end()
    self.histogramLabel.setPixmap(pixmap)


def styleThreeDViews():
  """Microscopy-style 3D views: orthographic projection, black background, white scale bar.
  The scale bar is Slicer's view ruler (thin), whose length is accurate in orthographic mode."""
  try:
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
      return
    for index in range(layoutManager.threeDViewCount):
      viewNode = layoutManager.threeDWidget(index).mrmlViewNode()
      viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
      viewNode.SetBackgroundColor(0., 0., 0.)      # the background is a gradient between the two colours
      viewNode.SetBackgroundColor2(0., 0., 0.)
      viewNode.SetRulerType(slicer.vtkMRMLAbstractViewNode.RulerTypeThin)
      viewNode.SetRulerColor(slicer.vtkMRMLAbstractViewNode.RulerColorWhite)
  except Exception as e:
    logging.warning(f"Could not style the 3D view: {e}")


def fitOrthographicScale():
  """Give the 3D cameras an orthographic (parallel) scale that matches the data.

  The camera's parallel scale is separate from its perspective distance, and is not reset along
  with it. Microscopy images are often a fraction of a millimetre wide (a few hundred
  micrometres, in Slicer's millimetre units), so a scale left over from larger data shows them as
  a dot once the 3D view is switched to orthographic projection. Like vtkRenderer.ResetCamera,
  this sets the scale (half the view height) to the radius of the bounding sphere of all volumes
  in the scene. It is changed on the MRML camera node, as Slicer requires."""
  import math
  try:
    lower, upper = [math.inf] * 3, [-math.inf] * 3
    for className in ('vtkMRMLScalarVolumeNode', 'vtkMRMLVectorVolumeNode'):
      for node in slicer.util.getNodesByClass(className):
        bounds = [0.] * 6
        node.GetRASBounds(bounds)
        for axis in range(3):
          lower[axis] = min(lower[axis], bounds[2 * axis])
          upper[axis] = max(upper[axis], bounds[2 * axis + 1])
    radius = 0.5 * math.sqrt(sum((u - l) ** 2 for l, u in zip(lower, upper)))
    if not math.isfinite(radius) or radius <= 0:
      return
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
      return
    camerasLogic = slicer.modules.cameras.logic()
    for index in range(layoutManager.threeDViewCount):
      viewNode = layoutManager.threeDWidget(index).mrmlViewNode()
      cameraNode = camerasLogic.GetViewActiveCameraNode(viewNode)
      if cameraNode is not None:
        cameraNode.SetParallelScale(radius)
  except Exception as e:
    logging.warning(f"Could not set the orthographic scale of the 3D view: {e}")


#
# Reader plugin
# (identified by its special name <moduleName>FileReader)
#

class ImportMicroscopyFileReader(object):

  def __init__(self, parent):
    self.parent = parent

  def description(self):
    return 'Microscopy image'

  def fileType(self):
    return 'MicroscopyImageFile'

  def extensions(self):
    return [f"{description} ({' '.join('*' + s for s in suffixes)})"
            for description, suffixes in _allFormats()]

  def checkRequiredPythonPackages(self):
    if sys.version_info < (3, 12):
      slicer.util.errorDisplay("Importing microscopy files requires Python 3.12 or newer (Slicer 5.10 or newer).")
      return False

    if self._packagesAvailable():
      return True

    message = ("Importing microscopy files requires the 'napari-virtual-reader' and 'refkv' Python packages "
               "(plus zarr, dask and imagecodecs).\n\n"
               "Slicer's numpy version will be kept as it is.")
    if not slicer.util.confirmOkCancelDisplay(message + "\n\nClick OK to install them now (it may take several minutes)."):
      return False

    try:
      import numpy
      # 1. Everything that can be resolved normally. numpy is pinned to what Slicer ships, so pip
      #    either keeps it or fails loudly; it never silently upgrades Slicer's numpy.
      slicer.util.pip_install([
        "zarr>=3.4", "dask[array]>=2024.1",
        "imagecodecs>=2026.8.16", "imagecodecs-zarr>=2026.8.16",
        f"numpy=={numpy.__version__}",
      ])
      # 2. refkv declares a newer numpy than Slicer may ship, although it runs fine without it.
      #    The reader plugin declares napari and Qt as dependencies, which we must not install.
      slicer.util.pip_install(["--no-deps", "refkv>=0.2.0"])
      slicer.util.pip_install(["--no-deps", PLUGIN_REQUIREMENT])
    except Exception as e:
      slicer.util.errorDisplay(f"Installing the required Python packages failed:\n\n{e}")
      return False

    if not self._packagesAvailable():
      slicer.util.errorDisplay("Required Python packages have not been installed. Cannot import microscopy image.")
      return False
    return True

  def _packagesAvailable(self):
    try:
      import refkv.zarr, zarr, dask.array, numpy
      from napari_virtual_reader import read_images
      return True
    except ImportError:
      return False

  def canLoadFile(self, filePath):
    # Cheap on purpose: no parsing and no package installation prompt while Slicer is only
    # guessing which reader to use. Problems are reported by load().
    return str(filePath).lower().endswith(_supportedSuffixes())

  def canLoadFileConfidence(self, filePath):
    if not self.canLoadFile(filePath):
      return 0.0
    name = str(filePath).lower()
    specific = tuple(s for _, suffixes in FORMATS for s in suffixes)
    # Plain TIFF (opt-in) gets a low confidence so that Slicer's own reader wins by default
    return 0.6 if name.endswith(specific) else 0.3

  def _readLayers(self, filePath):
    from napari_virtual_reader import read_images
    with _napariShim():
      return read_images(str(filePath))

  def _confirmLarge(self, nbytes, maxBytes):
    if slicer.app.commandOptions().noMainWindow or slicer.app.testingEnabled():
      return True
    return slicer.util.confirmOkCancelDisplay(
      f"A single volume of this image needs about {nbytes / 2**20:.0f} MB, and no smaller "
      f"resolution level is stored in the file (limit: {maxBytes / 2**20:.0f} MB).\n\nLoad it anyway?")

  def _chooseScene(self, sceneKeys, scenes):
    """Ask which image to load when the file holds several (like ImageJ's series selector).
    Returns an index, 'all', or raises if the dialog is cancelled. Without a GUI, or when the
    'scene' property is given, nothing is asked: scene 0 is used."""
    if len(sceneKeys) < 2 or slicer.app.commandOptions().noMainWindow or slicer.app.testingEnabled():
      return 0
    items = [f"{i}: {describeScene(key, scenes[key])}" for i, key in enumerate(sceneKeys)]
    allItem = f"All ({len(items)} images)"
    choice = qt.QInputDialog.getItem(slicer.util.mainWindow(), "Select image",
                                     f"This file contains {len(items)} images. Load:", items + [allItem], 0, False)
    if not choice:
      raise RuntimeError("Loading cancelled")
    return 'all' if choice == allItem else items.index(choice)

  @staticmethod
  def _parseProperties(properties):
    scene = properties.get('scene', None)   # None: not specified, ask (GUI) or use 0
    if isinstance(scene, str):
      scene = 'all' if scene.strip().lower() == 'all' else int(scene)
    timepoint = properties.get('timepoint', None)
    if timepoint is not None and timepoint != '':
      timepoint = int(timepoint)
    else:
      timepoint = None
    maxSizeMB = float(properties.get('maxSizeMB', DEFAULT_MAX_SIZE_MB))
    resetViews = properties.get('resetViews', True)
    if isinstance(resetViews, str):
      resetViews = resetViews.strip().lower() not in ('false', '0', 'no', '')
    showDialog = properties.get('showDisplayAdjustment', True)
    if isinstance(showDialog, str):
      showDialog = showDialog.strip().lower() not in ('false', '0', 'no', '')
    style3D = properties.get('style3DView', True)
    if isinstance(style3D, str):
      style3D = style3D.strip().lower() not in ('false', '0', 'no', '')
    return scene, timepoint, maxSizeMB, bool(resetViews), bool(showDialog), bool(style3D)

  def load(self, properties):
    if not self.checkRequiredPythonPackages():
      return False

    createdNodes = []
    try:
      import numpy as np

      filePath = properties['fileName']
      scene, timepoint, maxSizeMB, resetViews, showDialog, style3D = self._parseProperties(properties)
      maxBytes = maxSizeMB * 2**20

      layers = self._readLayers(filePath)
      scenes = collectScenes(layers)
      sceneKeys = list(scenes.keys())
      if scene is None:
        scene = self._chooseScene(sceneKeys, scenes)
      if scene == 'all':
        selected = list(range(len(sceneKeys)))
      elif 0 <= scene < len(sceneKeys):
        selected = [scene]
      else:
        raise ValueError(f"Scene {scene} requested, but the file has {len(sceneKeys)} scene(s)")
      if len(sceneKeys) > 1 and scene != 'all':
        logging.info(f"{len(sceneKeys)} scenes found, loading scene {scene}. "
                     "Use the 'scene' property (an index, or 'all') to load others.")

      if 'name' in properties.keys() and properties['name']:
        baseName = properties['name']
      else:
        baseName = Path(filePath).name

      confirmedLarge = False
      channelSettings = {}     # (scene, channel) -> shared rendering settings
      groupIds = {}            # scene -> id used to find its volumes again
      for sceneIndex in selected:
        entries = scenes[sceneKeys[sceneIndex]]
        nTimepoints = timepointCount(entries[0])
        if timepoint is not None and not 0 <= timepoint < nTimepoints:
          raise ValueError(f"Timepoint {timepoint} requested, but the image has {nTimepoints}")
        timepoints = [timepoint] if timepoint is not None else range(nTimepoints)

        for t in timepoints:
          for entry in entries:
            levelIndex, nbytes, fits = planEntry(entry, maxBytes)
            if levelIndex != 0:
              logging.info(f"Using pyramid level {levelIndex} of {len(entry['levels'])} to stay under {maxSizeMB:.0f} MB")
            if not fits and not confirmedLarge:
              if not self._confirmLarge(nbytes, maxBytes):
                raise RuntimeError("Loading cancelled")
              confirmedLarge = True

            name = baseName
            if len(sceneKeys) > 1:
              # several images in the file: say which one this volume comes from
              name += f" - {entry.get('series_name') or 'scene' + str(sceneIndex)}"
            if nTimepoints > 1 and timepoint is None:
              name += f" - tp{t}"
            name += f" - {entry['label']}"
            name = slicer.mrmlScene.GenerateUniqueName(name)

            voxels = normalizeDtype(readVoxels(entry, levelIndex, t))
            spacing, origin = geometry(entry, levelIndex)

            if entry["rgb"]:
              voxels = toRgb8(voxels, entry["limits"])
              volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLVectorVolumeNode', name)
            else:
              volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', name)
            createdNodes.append(volumeNode)

            volumeNode.SetSpacing(*spacing)
            volumeNode.SetOrigin(*origin)
            slicer.util.updateVolumeFromArray(volumeNode, voxels)
            volumeNode.CreateDefaultDisplayNodes()

            if not entry["rgb"] and voxels.shape[0] > 1:
              key = (sceneIndex, entry["channel"])
              if key not in channelSettings:
                # starting values from the first timepoint; all timepoints share them
                low, high = estimateOpacityRamp(sampleVoxels(voxels), float(voxels.min()), float(voxels.max()))
                channelSettings[key] = {"color": entry["color"] or [1., 1., 1.], "min": low, "max": high,
                                        "gamma": 1.0, "visible": True}
              groupId = groupIds.setdefault(sceneIndex, uuid.uuid4().hex)
              volumeNode.SetAttribute(ATTR + "group", groupId)
              volumeNode.SetAttribute(ATTR + "groupLabel", baseName + (
                f" - {entry.get('series_name') or 'scene' + str(sceneIndex)}" if len(sceneKeys) > 1 else ""))
              volumeNode.SetAttribute(ATTR + "channel", str(entry["channel"]))
              volumeNode.SetAttribute(ATTR + "label", entry["label"])
              volumeNode.SetAttribute(ATTR + "timepoint", str(t))
              applyRendering(volumeNode, channelSettings[key])
              saveSettings(volumeNode, channelSettings[key])

        if sceneIndex == selected[0]:
          # Slice views: first channel as background, second (if any) blended on top
          for position, node in enumerate(createdNodes[:2]):
            if position == 0:
              slicer.util.setSliceViewerLayers(background=node, fit=resetViews)
            else:
              slicer.util.setSliceViewerLayers(foreground=node, foregroundOpacity=0.5)

      if not createdNodes:
        raise ValueError("No image data found")
      if style3D:
        styleThreeDViews()
      if resetViews or style3D:     # orthographic projection needs the camera re-centred and re-scaled
        slicer.util.resetThreeDViews()
        fitOrthographicScale()

    except Exception as e:
      logging.error('Failed to load file: ' + str(e))
      import traceback
      traceback.print_exc()
      for node in createdNodes:
        slicer.mrmlScene.RemoveNode(node)
      if "cancelled" not in str(e):
        slicer.util.errorDisplay(f"Failed to load microscopy image:\n\n{e}")
      return False

    # Show volume
    selectionNode = slicer.app.applicationLogic().GetSelectionNode()
    selectionNode.SetActiveVolumeID(createdNodes[0].GetID())
    slicer.app.applicationLogic().PropagateVolumeSelection()

    self.parent.loadedNodes = [node.GetID() for node in createdNodes]

    if showDialog and groupIds and not slicer.app.commandOptions().noMainWindow and not slicer.app.testingEnabled():
      try:
        showDisplayAdjustment(groupIds[min(groupIds)])
      except Exception as e:
        logging.warning(f"Could not open the display adjustment dialog: {e}")
    return True


#
# ImportMicroscopyTest
#

class ImportMicroscopyTest(ScriptedLoadableModuleTest):

  def setUp(self):
    slicer.mrmlScene.Clear()

  def runTest(self):
    self.setUp()
    self.test_ImportMicroscopy1()

  def test_ImportMicroscopy1(self):
    """Write a small 2-timepoint, 2-channel OME-TIFF z-stack and load it back."""
    try:
      import refkv.zarr, zarr, dask.array
      from napari_virtual_reader import read_images
    except ImportError:
      self.delayDisplay("Required packages not installed, skipping test")
      return
    try:
      import tifffile
    except ImportError:
      slicer.util.pip_install("tifffile")
      import tifffile
    import numpy as np
    import tempfile

    self.delayDisplay("Creating test image")
    data = (np.random.default_rng(0).random((2, 4, 2, 16, 20)) * 4000).astype(np.uint16)  # T Z C Y X
    path = os.path.join(tempfile.mkdtemp(), "test.ome.tif")
    tifffile.imwrite(path, data, ome=True, photometric="minisblack", metadata=dict(
      axes="TZCYX", PhysicalSizeX=0.5, PhysicalSizeY=0.5, PhysicalSizeZ=2.0,
      PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm", PhysicalSizeZUnit="µm",
      Channel=dict(Name=["DAPI", "GFP"])))

    self.delayDisplay("Loading test image")
    self.assertTrue(slicer.util.loadNodeFromFile(path, 'MicroscopyImageFile'))

    nodes = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')
    self.assertEqual(len(nodes), 4)  # 2 timepoints x 2 channels

    self.delayDisplay('Checking loaded image')
    node = nodes[0]
    self.assertEqual(node.GetImageData().GetDimensions(), (20, 16, 4))
    for actual, expected in zip(node.GetSpacing(), (0.5e-3, 0.5e-3, 2e-3)):
      self.assertAlmostEqual(actual, expected, places=9)

    self.delayDisplay('Test passed')