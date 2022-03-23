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
