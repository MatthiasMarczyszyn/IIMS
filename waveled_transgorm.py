import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
import pywt
import pywt.data
import cv2
# Load image


def DWT(data, max_level):
    LL=data
    for level in range(max_level):
        new_data = LL
        size=LL[1].size/2
        size = int(size)
        LL = np.zeros((size,size))
        LH = np.zeros((size,size))
        HL = np.zeros((size,size))
        HH = np.zeros((size,size))
        for x in range(0,size*2,2):
           for y in range(0,size*2,2):
                LL[x//2][y//2] = (new_data[x][y] + new_data[x][y+1] + new_data[x+1][y] + new_data[x+1][y+1])/4
                LH[x//2][y//2] = (new_data[x][y] + new_data[x][y+1] - new_data[x+1][y] - new_data[x+1][y+1])/4
                HL[x//2][y//2] = (new_data[x][y] - new_data[x][y+1] + new_data[x+1][y] - new_data[x+1][y+1])/4
                HH[x//2][y//2] = (new_data[x][y] - new_data[x][y+1] - new_data[x+1][y] + new_data[x+1][y+1])/4
        return LL, (LH,HL,HH)



img_name = "lena.jpeg"
original = cv2.imread(os.getcwd() + "/" + img_name,cv2.IMREAD_GRAYSCALE)
# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original,"haar")
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.savefig("result_" + img_name )