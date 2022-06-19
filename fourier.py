import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
import pywt
import pywt.data
import cv2


def highPassFiltering(img,size):#Transfer parameters are Fourier transform spectrogram and filter size
    h, w = img.shape[0:2]#Getting image properties
    h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0#Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 0
    return img

def lowPassFiltering(img,size):
    h, w = img.shape[0:2]  # Getting image properties
    h1, w1 = int(h / 2), int(w / 2)  # Find the center point of the Fourier spectrum
    img[0:h, 0:w1 - int(size / 2)] = 0
    img[0:h, w1 + int(size / 2):w] = 0
    img[0:h1 - int(size / 2), 0:w] = 0
    img[h1 + int(size / 2):h, 0:w] = 0
    return img

gray = cv2.imread("lena.jpeg", 1)
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

img_dft = np.fft.fft2(gray)
dft_shift = np.fft.fftshift(img_dft)  # Move frequency domain from upper left to middle

#High pass filter
dft_shift=highPassFiltering(dft_shift,0)
#dft_shift=lowPassFiltering(dft_shift,20)
res = np.log(np.abs(dft_shift))


# Inverse Fourier Transform
idft_shift = np.fft.ifftshift(dft_shift)  #Move the frequency domain from the middle to the upper left corner
ifimg = np.fft.ifft2(idft_shift)  # Fourier library function call
ifimg = np.abs(ifimg)
cv2.imshow("ifimg",ifimg)
cv2.imshow("gray",gray)


# Draw pictures
plt.subplot(131), plt.imshow(gray, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(132), plt.imshow(res, 'gray'), plt.title('High pass filter')
plt.axis('off')
plt.subplot(133), plt.imshow(ifimg, 'gray'), plt.title('Effect after filtering')
plt.axis('off')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()