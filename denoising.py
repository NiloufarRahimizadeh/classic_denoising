import matplotlib.pyplot as plt
import pydicom


fpath = pydicom.dcmread("/home/oem/Desktop/behpardaz/real_data/Ahmad_Swaid_CBCT/0000.dcm")

img = fpath.pixel_array
plt.imshow(img, cmap=plt.cm.bone)
# plt.show()
plt.imsave("dcm_to_tiff_converted.tiff", img, cmap='gray')

#BM3D Block-matching and 3D filtering
#pip install bm3d

import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio
import bm3d
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import cv2
from skimage.restoration import denoise_bilateral
from skimage.restoration import denoise_nl_means, estimate_sigma



noisy_img = img_as_float(io.imread("dcm_to_tiff_converted.tiff", as_gray=True))
# ref_img = img_as_float(io.imread("images/MRI_images/MRI_clean.tif"))
norm = (img - np.min(img)) / (np.max(img) - np.min(img))
# sigma_est = np.mean(estimate_sigma(img, multichannel=True))


# BM3D_denoised_img = bm3d.bm3d(norm, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
     
# TV_denoised_img = denoise_tv_chambolle(norm, weight=0.1, eps=0.0002, n_iter_max=50, multichannel=False)

bilateral_denoised_img = denoise_bilateral(norm, sigma_color=0.1, sigma_spatial=10,
                multichannel=False)


# NLM_denoised_img = denoise_nl_means(noisy_img, h=2 * sigma_est, fast_mode=True,
#                                patch_size=5, patch_distance=3, multichannel=False)

# noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
# BM3D_cleaned_psnr = peak_signal_noise_ratio(ref_img, BM3D_denoised_image)
# print("PSNR of input noisy image = ", noise_psnr)
# print("PSNR of cleaned image = ", BM3D_cleaned_psnr)

# plt.imsave("BM3D_denoised.tiff", BM3D_denoised_image, cmap='gray')
plt.imshow(bilateral_denoised_img, cmap='gray')
plt.show()