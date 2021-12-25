from pydicom import dcmread
import matplotlib.pyplot as plt

filename = "/home/oem/Desktop/behpardaz/real_data/Ahmad_Swaid_CBCT/0000.dcm"

ds = dcmread(filename)
arr = ds.pixel_array


img = ds.pixel_array
plt.imshow(img, cmap=plt.cm.bone)
plt.show()