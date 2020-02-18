import numpy as np
import cv2
import colorsys

img = cv2.imread('/home/iiitb/Desktop/watch.jpg', 0)     # read a image using imread 
equ = cv2.equalizeHist(img)								 # creating a Histograms Equalization 
														 # of a image using cv2.equalizeHist()

#res = np.hstack((img, equ))                              # stacking images side-by-side 
cv2.imshow('Histogram Equalization', equ)

cv2.waitKey(0)          
cv2.destroyAllWindows()