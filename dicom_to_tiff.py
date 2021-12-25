import os
import numpy as np

train_images=sorted(os.listdir('/home/oem/Desktop/Medical/denoising/src/head_image'))
# print(train_images)

import matplotlib.pyplot as plt
import pydicom
id=0
for im in train_images:
    fpath = pydicom.dcmread("/home/oem/Desktop/Medical/denoising/src/head_image/"+im)

    img = fpath.pixel_array
    # plt.imshow(img, cmap=plt.cm.bone)
    # plt.show()
    plt.imsave(f"/home/oem/Desktop/Medical/denoising/src/Head_clean/dcm_to_tiff_converted{id}.tiff", img, cmap='gray')
    id = id + 1