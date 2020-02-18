import numpy as np
import cv2
import colorsys

img = cv2.imread('/home/iiitb/Desktop/watch.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray_Scale_Image', gray)

cv2.waitKey(0)          
cv2.destroyAllWindows()
print(img.shape)