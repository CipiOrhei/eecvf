"""
Camera calibration using a checkerboard pattern
"""
##
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ex. 1 -> Load the sequence of chessboard images
# The directory chess1 contains a set of images left01.jpg, left02.jpg, ... left14.jpg that can be used for camera calibration.
# Use a glob pattern to create a list images of all the image filepaths (i.e., a list of strings).
# ---

# ---


## Ex. 2 -> Examine the first image
# Extract the first image filename from the list of images and convert it to grayscale
# hints: cv2.imread, cv2.cvtColor, plt.imshow (gray)
# ---

# ---

## Ex. 3 -> Find corners of the chessboard and plot them on the image
# hints: cv2.findChessboardCorners, np.squeeze,
# Use cv2.drawChessboardCorners to plot the detected points
# Assume a pattern of size of (9,6)
# Use NumPy's squeeze function to eliminate singleton dimensions from the array corners
# ---

# ---

## Ex. 4 (optional) Refine corners to subpixel
# Hints: cv2.cornerSubPix
# The cornerSubPix function from OpenCV can be used to refine the corners extracted to sub-pixel accuracy.
# This is based on an iterative technique
# as such, one of the inputs criteria uses a tuple to bundle a convergence tolerance and a maximum number of iterations.
# ---

# ---

## Ex. 5 Apply operations to all images and perform calibration
# Hints: Use cv2.calibrateCamera to calibrate based on the accumulated object and image points
# ---

# ---

## Ex. 6 - Apply determined distortion matrix to correct and an image from chess1 folder
# Hints: cv2.getOptimalNewCameraMatrix, cv2.undistort
# ---

# ---

## Ex. 7 - Compute reprojection error for all points and write the mean error
# Hints: cv2.projectPoints, cv2.norm
# ---

# ---

