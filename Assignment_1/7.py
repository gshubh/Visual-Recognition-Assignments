import cv2
import numpy as np
from matplotlib import pyplot as plt

def salt_n_pepper(img, pad = 101, show = 1):
	img = to_std_float(img)    
	print(img.shape)      		  # Convert img1 to 0 to 1 float to avoid wrapping that occurs with uint8
	noise = np.random.randint(pad, size = (img.shape[0], img.shape[1], 1))
	img = np.where(noise == 0, 0, img)		  # Convert high and low bounds of pad in noise to salt and pepper noise then add it to.
	img = np.where(noise == (pad-1), 1, img)  # Our image. 1 is subtracted from pad to match bounds behaviour of np.random.randint.
	img = to_std_uint8(img)    				  # conversion from float16 back to uint8
	return img


def to_std_float(img):
	img.astype(np.float16, copy = False)
	img = np.multiply(img, (1/255))
	return img


def to_std_uint8(img):
	img = cv2.convertScaleAbs(img, alpha = (255/1)) 
	return img


def main():
	img = cv2.imread('/home/iiitb/Desktop/watch.jpg', 0) 
	equ = cv2.equalizeHist(img)
	gausBlur = cv2.GaussianBlur(equ, (15,15), 0)
	salt_and_pepper_image = salt_n_pepper(gausBlur, 101, 1)
	median = cv2.medianBlur(img,5)

	plt.subplot(121),plt.imshow(gausBlur),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(salt_and_pepper_image),plt.title('salt_and_pepper_image')
	plt.xticks([]), plt.yticks([])
	plt.subplot(123),plt.imshow(median),plt.title('Median_filtered_image')
	plt.xticks([]), plt.yticks([])
	plt.show()
	return 0

if __name__ == '__main__':	
 	main()


