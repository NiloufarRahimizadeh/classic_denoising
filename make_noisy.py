import os
import numpy as np

train_images=sorted(os.listdir('/home/oem/Desktop/Medical/denoising/src/Dental_clean'))
# print(train_images)

from keras.preprocessing import image
train_image=[]
for im in train_images:
    img=image.load_img("/home/oem/Desktop/Medical/denoising/src/Dental_clean/" + im,target_size=(64, 64),color_mode='grayscale')
    img=image.img_to_array(img)
    img=img/255
    train_image.append(img)

train_df=np.array(train_image)
# print(train_df.shape)

def add_noise(image):
    row,col,ch=image.shape
    mean=0
    sigma=1
    gauss=np.random.normal(mean,sigma,(row,col,ch))
    gauss=gauss.reshape(row,col,ch)
    noisy=image+gauss*0.05
    return noisy

noised_df=[]
for img in train_df:
  noisy=add_noise(img)
  noised_df.append(noisy)

noised_df=np.array(noised_df)


# saving noisy images
import matplotlib.pyplot as plt

id=0
for im in range(noised_df.shape[0]):
    img =  noised_df[im, :, :]

    # plt.imshow(img.reshape(128,128), cmap='gray')
    plt.imsave(f"/home/oem/Desktop/Medical/denoising/src/Dental_noisy/dcm_to_tiff_converted{id}.tiff", img.reshape(64, 64), cmap='gray')
    id = id + 1