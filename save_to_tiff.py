import os
import numpy as np

train_images=sorted(os.listdir('/home/oem/Desktop/Medical/denoising/src/dental'))
# print(train_images)

from keras.preprocessing import image
train_image=[]
for im in train_images:
    img=image.load_img("/home/oem/Desktop/Medical/denoising/src/dental/" + im,target_size=(64, 64),color_mode='grayscale')
    img=image.img_to_array(img)
    img=img/255
    train_image.append(img)

train_df=np.array(train_image)
# print(train_df.shape)


# saving noisy images
import matplotlib.pyplot as plt

id=0
for im in range(train_df.shape[0]):
    img =  train_df[im, :, :]

    # plt.imshow(img.reshape(128,128), cmap='gray')
    plt.imsave(f"/home/oem/Desktop/Medical/denoising/src/Dental_clean/dcm_to_tiff_converted{id}.tiff", img.reshape(64, 64), cmap='gray')
    id = id + 1