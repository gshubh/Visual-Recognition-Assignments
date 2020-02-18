
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


# In[2]:


img1 = cv2.imread("institute1.jpg",1)           # FIRST PART OF INSTITUTE
copy_img1 = cv2.imread("institute1.jpg",1)
img2 = cv2.imread("institute2.jpg",1)           # SECOND PART OF THE INSTITUTE
copy_img_2 = cv2.imread("institute2.jpg",1)
img3 = cv2.imread("secondPic1.jpg",1)           # FIRST PART OF OUTSIDE LIBRARY
copy_img3 = cv2.imread("secondPic1.jpg",1)
img4 = cv2.imread("secondPic2.jpg",1)           # SECOND PART OF OUTSIDE LIBRARY
copy_img4 = cv2.imread("secondPic2.jpg",1)

# "sift" function is used to find the key point and descriptor of image,like nose, eyes, lips, eyebros etc in face image 
sift = cv2.xfeatures2d.SIFT_create()    

keypoints1,descriptors1 = sift.detectAndCompute(img1,None)
keypoints2,descriptors2 = sift.detectAndCompute(img2,None)
keypoints3,descriptors3 = sift.detectAndCompute(img3,None)
keypoints4,descriptors4 = sift.detectAndCompute(img4,None)


# Image that contain key points in the First part of Institute. 
img5 = cv2.drawKeypoints(copy_img1,keypoints1,copy_img1)
cv2.imshow("KPOI",img5)      
cv2.imwrite("KPOI1.jpg",img5)

# Image that contain key points in the Second part of Institute.
img6 = cv2.drawKeypoints(copy_img_2,keypoints2,copy_img_2)
cv2.imshow("KPOI",img6)
cv2.imwrite("KPOI2.jpg",img6)

# Image that contain key points in the Second part of Outside Library.
img7 = cv2.drawKeypoints(copy_img3,keypoints3,copy_img3)
cv2.imshow("KPOI",img7)
cv2.imwrite("KPOL1.jpg",img7)

# Image that contain key points in the Second part of Outside Library.
img8 = cv2.drawKeypoints(copy_img4,keypoints4,copy_img4)
cv2.imshow("KPOI",img8)
cv2.imwrite("KPOL2.jpg",img8)


bruteforcematcher = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)    # Match the descriptor of two images and check the common part in two images

imagematch1 = bruteforcematcher.match(descriptors1,descriptors2)  # match the descriptor of two images of the institute.
imagematch2 = bruteforcematcher.match(descriptors3,descriptors4)  # match the descriptor of two images of the outside library.
imagematch3 = sorted(imagematch1,key=lambda x:x.distance)         # Sorting of the descriptors on the basis of distance 
imagematch4 = sorted(imagematch2,key=lambda x:x.distance)         # Sorting of the descriptors on the basis of distance

match_img1 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,imagematch3[:150],None,flags=2)  # Join the match points of image 1 and image 2 by a straight line
match_img2 = cv2.drawMatches(img3,keypoints3,img4,keypoints4,imagematch4[:150],None,flags=2)  # Join the match points of image 3 and image 4 by a straight line


cv2.imshow("INSTITUTE IMAGE AFTER MATCHING POINTS",match_img1)
cv2.imwrite("match_img_1.jpg",match_img1)
cv2.imshow("OUTSIDE LIBRARY IMAGE AFTER MATCHING POINTS",match_img2)
cv2.imwrite("match_img_2.jpg",match_img2)

src_pts1 = np.float32([keypoints1[m.queryIdx].pt for m in imagematch3]).reshape(-1,1,2) # fill the common point of both images in array
des_pts1 = np.float32([keypoints2[m.trainIdx].pt for m in imagematch3]).reshape(-1,1,2)
src_pts2 = np.float32([keypoints3[m.queryIdx].pt for m in imagematch4]).reshape(-1,1,2)
des_pts2 = np.float32([keypoints4[m.trainIdx].pt for m in imagematch4]).reshape(-1,1,2)

matrix1,_=cv2.findHomography(src_pts1,des_pts1,cv2.RANSAC,5.0)  # findHomography function give the transformation matrix of transformation bw src_pts1 and des_pts1.
matrix2,_=cv2.findHomography(src_pts2,des_pts2,cv2.RANSAC,5.0)  # findHomography function give the transformation matrix of transformation bw src_pts2 and des_pts2.


# To apply perspective transformation to both images, "warpPerspective" transforms the source image using the specified matrix.
img9 = cv2.warpPerspective(img2,matrix1,(img1.shape[1]+img2.shape[1],img1.shape[0]))   # img9 is WARPED IMAGE OF INSTITUTE.
img10 = cv2.warpPerspective(img4,matrix2,(img3.shape[1]+img4.shape[1],img3.shape[0]))  # img10 is WARPED IMAGE OF OUTSIDE LIBRARY.


copy_img1 = np.concatenate([img1,img9],axis=1)  # Concatenation of 1st image of institute(img1) and WARPED IMAGE OF INSTITUTE.
img12 = np.concatenate([img3,img10],axis=1) # Concatenation of 1st image of outside library(img3) and WARPED IMAGE OF OUTSIDE LIBRARY.

cv2.imshow("PANORAMA OF INSTITUTE",copy_img1)
cv2.imwrite("pano1.jpg",copy_img1)
cv2.imshow("PANORAMA OF OUTSIDE LIBRARY",img12)
cv2.imwrite("pano2.jpg",img12)
cv2.waitKey(0)
cv2.destroyAllWindows()        


#  
