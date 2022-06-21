from scipy.fftpack import dct, idct
import math
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')
# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


def dct2_c(a):
    k=0
    imd=np.ndarray(shape=(512,512), dtype=float, order='F')
    for row in range(512):
        for index in range(512):
            for i in range(512):
                imd[row,index] = imd[row,index]+a[row,i] * math.cos((math.pi / 512) * index * (i + 1 / 2))

    return imd

def idct2_c(a):
    k=0
    imdi=np.ndarray(shape=(512,512), dtype=float, order='F')
    for row in range(512):
        for index in range(512):

            for i in range(512):
                imdi[row,index] = imdi[row,index]+a[row,i] * math.cos((math.pi / 512) * i * (index + 1 / 2))
                if(i==511):
                    imdi[row, index]=imdi[row,index]+(1/2)*a[row,0]
    return imdi

# read lena RGB image and convert to grayscale
im = rgb2gray(imread('Images/lena.jpg'))
print(len(im))
imF = dct2(im)
im1 = idct2(imF)
image_dct=dct2_c(im)
iImage=idct2_c(image_dct)
# print(image_dct[0,:5])
# print(imF[0,:5])
#
# check if the reconstructed image is nearly equal to the original image
print(np.allclose(im, im1))
# True

# plot original and reconstructed images with matplotlib.pylab
plt.gray()
plt.subplot(221), plt.imshow(im), plt.axis('off'), plt.title('original image', size=20)
plt.subplot(222), plt.imshow(iImage), plt.axis('off'), plt.title('reconstructed image (DCT+IDCT)', size=20)
plt.subplot(223), plt.imshow(im), plt.axis('off'), plt.title('original image', size=20)
plt.subplot(224), plt.imshow(im1), plt.axis('off'), plt.title('reconstructed image with scipy library(DCT+IDCT)', size=20)
plt.show()
