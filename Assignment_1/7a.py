import numpy as np
import random
import cv2
from matplotlib import pyplot as plt


def sp_noise(image,prob):   #Add salt and pepper noise to image, prob: Probability of the noise.
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

img = cv2.imread('/home/iiitb/Desktop/watch.jpg', 0) 
equ = cv2.equalizeHist(img)
gausBlur = cv2.GaussianBlur(equ, (15,15), 0)
noise_img = sp_noise(gausBlur,0.05)
median = cv2.medianBlur(noise_img,5)


plt.subplot(121),plt.imshow(gausBlur),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(121),plt.imshow(noise_img),plt.title('salt_and_pepper_image')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Median_filtered_image')
plt.xticks([]), plt.yticks([])
plt.show()
