import vtk
import numpy as np


PathDicom = "/home/oem/Desktop/behpardaz/real_data/Ahmad_Swaid_CBCT"
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(PathDicom)
reader.Update()

_extent = reader.GetDataExtent()
ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
ConstPixelSpacing = reader.GetPixelSpacing()

shiftScale = vtk.vtkImageShiftScale()
shiftScale.SetScale(reader.GetRescaleSlope())
shiftScale.SetShift(reader.GetRescaleOffset())
shiftScale.SetInputConnection(reader.GetOutputPort())
shiftScale.Update()

threshold = vtk.vtkImageThreshold ()
threshold.SetInputConnection(reader.GetOutputPort())
threshold.ThresholdByLower(400)  # remove all soft tissue
threshold.ReplaceInOn()
threshold.SetInValue(0)  # set all values below 400 to 0
threshold.ReplaceOutOn()
threshold.SetOutValue(1)  # set all values above 400 to 1
threshold.Update()

dmc = vtk.vtkDiscreteMarchingCubes()
dmc.SetInputConnection(threshold.GetOutputPort())
dmc.GenerateValues(1, 1, 1)
dmc.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(dmc.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1.0, 1.0, 1.0)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
renWin.SetSize(300, 300)

camera = renderer.MakeCamera()
camera.SetPosition(-500.0, 245.5, 122.0)
camera.SetFocalPoint(301.0, 245.5, 122.0)
camera.SetViewAngle(30.0)
camera.SetRoll(-90.0)
renderer.SetActiveCamera(camera)
# vtk_show(renderer, 600, 600)

# stl
# writer = vtk.vtkSTLWriter()
# writer.SetInputConnection(dmc.GetOutputPort())
# writer.SetFileTypeToBinary()
# writer.SetFileName("threeD.stl")
# writer.Write()

# gltf
# exporter = vtk.vtkGLTFExporter()
# exporter.SetRenderWindow(renWin)
# exporter.SetFileName("mymodel.gltf")
# exporter.Write()

# obj
exporter = vtk.vtkOBJExporter()
exporter.SetFilePrefix('head')
exporter.SetInput(renWin)
exporter.Write()  

