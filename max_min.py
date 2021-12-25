from keras.preprocessing import image
from numpy.lib.type_check import imag
from sklearn.preprocessing import MinMaxScaler
import numpy as np

img=image.load_img("/home/oem/Desktop/Medical/denoising/src/Head_clean/dcm_to_tiff_converted1.tiff",target_size=(512, 512),color_mode='grayscale')
img=np.array(img)
scaler = MinMaxScaler()
# print(img.min())
# print(img.max())
# print(img.shape)
scaler.fit(img)
scaler.data_max_
print(scaler.transform(img))

