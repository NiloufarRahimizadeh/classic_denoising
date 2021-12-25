import matplotlib.pyplot as plt
import pydicom


fpath = pydicom.dcmread("temp.dcm")

img = fpath.pixel_array
img.m
plt.imshow(img, cmap=plt.cm.bone)
plt.show()