import numpy as np
import cv2
import colorsys
import math
img = cv2.imread('/home/iiitb/Desktop/watch.jpg')
r,g,b = cv2.split(img)

HSV_IMAGE = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(HSV_IMAGE)
cv2.imshow('Hue_image1', h)
cv2.imshow('Saturation_image1', s)
cv2.imshow('Variance_image', v)



img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
h,l,s = cv2.split(img_hls)

cv2.imshow('Hue_Image', h)
cv2.imshow('Saturation_Image', s)
cv2.imshow('lightness_Image', l)

cv2.waitKey(0)          
cv2.destroyAllWindows() 