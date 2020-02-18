import numpy as np
import cv2
img = cv2.imread('/home/iiitb/Desktop/watch.jpg')
b,g,r = cv2.split(img)
cv2.imshow('Red_image', r)
cv2.imshow('Green_image', g)
cv2.imshow('Blue_image', b)
cv2.waitKey(0)          
cv2.destroyAllWindows() 
