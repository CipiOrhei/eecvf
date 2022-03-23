"""
Bird eye view projection with homography matrix
Select 4 source points and 4 destination points in the image
and use opencv functions to compute the perspective transformation
between the two sets of points to obtain a birdeye view projection
"""
##
import matplotlib.pyplot as plt
import cv2
import numpy as np

## Ex 1 - Read bike.jpg and convert to RGB
# Select 4 points in the image and convert resulting array to numpy array of type np.float32 - use plt.ginput()
# Make sure that destination points are in the same order as the input points
# Use cv2.getPerspectiveTransform to get the homography matrix transformation
# Use cv2.warpPerspective to transform the input image
# Show the transformed image
# ---

# ---

# Solution 1
img = cv2.cvtColor(cv2.imread('bike.jpg'), cv2.COLOR_BGR2RGB)
w, h, c = img.shape
# select 4 points
plt.imshow(img)
points = plt.ginput(4)
src_points = np.array(points, np.float32)
print(src_points)

dst_points = np.array([(w/4, h/4), (w/4, 3*h/4), (3*w/4, 3*h/4), (3*w/4, h/4)], np.float32)

M = cv2.getPerspectiveTransform(src_points, dst_points)
print(M)
res = cv2.warpPerspective(img, M, (w, h), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

plt.imshow(res)
