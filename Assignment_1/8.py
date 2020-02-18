import numpy as np
import random
import cv2
import math
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

image = cv2.imread('/home/iiitb/Desktop/road4.jpg') 
height = image.shape[0]
width = image.shape[1]

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    # Create a match color with the same color channel counts.
    #match_mask_color = (255,) * channel_count
    match_mask_color = 255
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    # If there are no lines to draw, exit.
    if lines is None:
        return img
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img



region_of_interest_vertices = [(0, height), (width / 2, height / 2), (width, height),]

# Convert to grayscale here.
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cannyed_image = cv2.Canny(gray_image, 100, 200)
cropped_image = region_of_interest(
    cannyed_image,
    np.array(
        [region_of_interest_vertices],
        np.int32
    ),
)

lines = cv2.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)

left_line_x = []
left_line_y = []
right_line_x = []
right_line_y = []

left_line_x = []
left_line_y = []
right_line_x = []
right_line_y = []

if lines is not None:
	for line in lines:
	    for x1, y1, x2, y2 in line:
	        slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
	        if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
	            continue
	        if slope <= 0: # <-- If the slope is negative, left group.
	            left_line_x.extend([x1, x2])
	            left_line_y.extend([y1, y2])
	        else: # <-- Otherwise, right group.
	            right_line_x.extend([x1, x2])
	            right_line_y.extend([y1, y2])
min_y = image.shape[0] * (3 / 5) # <-- Just below the horizon
max_y = image.shape[0] # <-- The bottom of the image
poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1, rcond=None, full=False, w=None, cov=False
))
left_x_start = int(poly_left(max_y))
left_x_end = int(poly_left(min_y))
poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1, rcond=None, full=False, w=None, cov=False
))
right_x_start = int(poly_right(max_y))
right_x_end = int(poly_right(min_y))

line_image = draw_lines(
    image,
    [[
        [int(left_x_start), int(max_y), int(left_x_end), int(min_y)],
        [int(right_x_start), int(max_y), int(right_x_end), int(min_y)],
    ]], 
    color = [255,0,0],
    thickness=5
)
plt.figure()
plt.imshow(line_image)
plt.show()

