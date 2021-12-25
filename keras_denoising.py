import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
train_images=sorted(os.listdir('/home/oem/Desktop/behpardaz/DECADA(Dental Design CAD+AI)/Ahmad Swaid CBCT/'))
# print(train_images)


##########################################
import matplotlib.pyplot as plt
# import pydicom
# id=0
# for im in train_images:
#     fpath = pydicom.dcmread("/home/oem/Desktop/behpardaz/DECADA(Dental Design CAD+AI)/Ahmad Swaid CBCT/"+im)

#     img = fpath.pixel_array
#     # plt.imshow(img, cmap=plt.cm.bone)
#     # plt.show()
#     plt.imsave(f"/home/oem/Desktop/Medical/denoising/src/head_image/dcm_to_tiff_converted{id}.tiff", img, cmap='gray')
#     id = id + 1

###########################################

train_images=sorted(os.listdir('/home/oem/Desktop/Medical/denoising/src/head_image/'))
from keras.preprocessing import image
train_image=[]
for im in train_images:
  img=image.load_img("/home/oem/Desktop/Medical/denoising/src/head_image/" + im,target_size=(512, 512),color_mode='grayscale')
  img=image.img_to_array(img)
  img=img/255
  train_image.append(img)

train_df=np.array(train_image)

#Subplotting images
def plot_img(dataset):
  f,ax=plt.subplots(1,5)
  f.set_size_inches(40,20)
  for i in range(5,10):
    ax[i-5].imshow(dataset[i].reshape(512,512), cmap='gray')
  plt.show()

# plot_img(train_df)

#Adding gaussian noise with 0.05 factor
def add_noise(image):
  row,col,ch=image.shape
  mean=0
  sigma=1
  gauss=np.random.normal(mean,sigma,(row,col,ch))
  gauss=gauss.reshape(row,col,ch)
  noisy=image+gauss*0.1
  return noisy

noised_df=[]
for img in train_df:
  noisy=add_noise(img)
  noised_df.append(noisy)

noised_df=np.array(noised_df)
# plot_img(noised_df)

# xnoised=noised_df[:200]
# xtest=noised_df[200:]
xnoised=train_df[:200]
xtest=train_df[200:]

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D,MaxPool2D ,UpSampling2D, Flatten, Input
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras import backend as K

def autoencoder():

    input_img=Input(shape=(512,512,1),name='image_input')
    #enoder 
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='Conv1')(input_img)
    x = MaxPooling2D((2,2), padding='same', name='pool1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='Conv2')(x)
    x = MaxPooling2D((2,2), padding='same', name='pool2')(x)

    #decoder
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='Conv3')(x)
    x = UpSampling2D((2,2), name='upsample1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='Conv4')(x)
    x = UpSampling2D((2,2), name='upsample2')(x)
    x = Conv2D(1, (3,3), activation='sigmoid', padding='same', name='Conv5')(x)

    #model
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

model= autoencoder()
model.summary()

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

with tf.device('/device:GPU:0'):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    model.fit(xnoised, xnoised, epochs=40, batch_size=10, validation_data=(xtest, xtest), callbacks=[early_stopping])