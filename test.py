import matplotlib.pyplot as plt
from skimage import io, img_as_float



noisy_img = img_as_float(io.imread("/home/oem/Downloads/out.tiff", as_gray=True))

plt.imshow(noisy_img, cmap=plt.cm.bone)
plt.show()
