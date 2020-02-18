import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/iiitb/Desktop/watch.jpg',0) 
equ = cv2.equalizeHist(img)	
 
gausBlur = cv2.GaussianBlur(equ, (5,5), 0)				  #  LPF helps in removing noise, or blurring the image
                                                              #  A HPF filters helps in finding edges in an image.

res = np.hstack((equ, gausBlur))                              # stacking images side-by-side

cv2.imshow('Guassian Smoothning', res)

cv2.waitKey(0)          
cv2.destroyAllWindows()

#   Guassian kernel should be positive and odd
#   Gaussian filtering is highly effective in removing Gaussian noise from the image. And with the increase in scale 
#   basically increase in height and width of Kernal Image smoothened alot or leads to Increase in Image Blurring.