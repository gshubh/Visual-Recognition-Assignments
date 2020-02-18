import numpy as np
import cv2
import colorsys

img = cv2.imread('/home/iiitb/Desktop/watch.jpg')
r,g,b = cv2.split(img)


lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)

l = lab[:,:,0]
a = lab[:,:,1]
b = lab[:,:,2]

cv2.imshow('L_image', l)
cv2.imshow('a_image', a)
cv2.imshow('b_image', b)

cv2.waitKey(0)          
cv2.destroyAllWindows() 