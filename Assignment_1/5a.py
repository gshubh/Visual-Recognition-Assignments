import numpy as np
import cv2
import colorsys
import math

img = cv2.imread('/home/iiitb/Desktop/watch.jpg')
X = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def image_whitening(image):
    new_image=np.zeros(shape=(image.shape[0],image.shape[1]))
    mean_image=image.mean()
    variance=image.var()
    standard_deviation=math.sqrt(variance)    
    new_image = (image-mean_image)/standard_deviation
    mini = np.min(new_image)   
    maxi = np.max(new_image)
    new_image = (new_image-mini)/(maxi-mini)
    return new_image

final_image = image_whitening(X)
cv2.imshow('Whitened_Image', final_image)
cv2.imshow('Grayscale_Image', X)
cv2.waitKey(0)          
cv2.destroyAllWindows()


