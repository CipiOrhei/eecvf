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
##
# Solution 1
import glob
pattern_size = (9, 6)
images = sorted(glob.glob("chess1/left??.jpg"))
print(images)

## Ex. 2 -> Examine the first image
# Extract the first image filename from the list of images and convert it to grayscale
# hints: cv2.imread, cv2.cvtColor, plt.imshow (gray)
# ---

# ---

## Solution 2
img = cv2.imread(images[0])
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape, gray.shape)
plt.imshow(gray, cmap='gray')

## Ex. 3 -> Find corners of the chessboard and plot them on the image
# hints: cv2.findChessboardCorners, np.squeeze,
# Use cv2.drawChessboardCorners to plot the detected points
# Assume a pattern of size of (9,6)
# Use NumPy's squeeze function to eliminate singleton dimensions from the array corners
# ---

# ---

## Solution 3
gray = cv2.equalizeHist(gray)
retval, corners = cv2.findChessboardCorners(image=gray, patternSize=pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                                  # cv2.CALIB_CB_FAST_CHECK +
                                                                                  cv2.CALIB_CB_NORMALIZE_IMAGE)
if retval:
    print(corners)
    corners = np.squeeze(corners)  # Get rid of extraneous singleton dimension
    print(corners.shape)
    print(corners[:5])  # Examine the first few rows of corners

## Ex. 4 (optional) Refine corners to subpixel
# Hints: cv2.cornerSubPix
# The cornerSubPix function from OpenCV can be used to refine the corners extracted to sub-pixel accuracy.
# This is based on an iterative technique
# as such, one of the inputs criteria uses a tuple to bundle a convergence tolerance and a maximum number of iterations.
# ---

# ---

# Solution 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_orig = corners.copy()
corners = cv2.cornerSubPix(gray, corners, pattern_size, (-1, -1), criteria=criteria)

## Plot refined corners
img3 = np.copy(img)

img4 = cv2.drawChessboardCorners(img, pattern_size, corners, retval)
plt.figure(figsize=(10, 10))
plt.imshow(img4)

## Ex. 5 Apply operations to all images and perform calibration
# Hints: Use cv2.calibrateCamera to calibrate based on the accumulated object and image points
# ---

# ---

# Solution 5
obj_grid = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
obj_grid[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

obj_points = []  # 3d world coordinates
img_points = []  # 2d image coordinates

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    print('Loading {}'.format(fname))
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    retval, corners = cv2.findChessboardCorners(image=gray, patternSize=pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                                        # cv2.CALIB_CB_FAST_CHECK +
                                                                                        cv2.CALIB_CB_NORMALIZE_IMAGE)
    if retval:
        obj_points.append(obj_grid)
        corners2 = cv2.cornerSubPix(gray, corners, pattern_size, (-1, -1), criteria)
        img_points.append(corners2)

retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print(retval)  # Objective function value
print(mtx)  # Camera matrix
print(dist)  # Distortion coefficients

## Ex. 6 - Apply determined distortion matrix to correct and an image from chess1 folder
# Hints: cv2.getOptimalNewCameraMatrix, cv2.undistort
# ---

# ---

# Solution 6
img = cv2.imread('chess1/left03.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(img)
plt.title('Original')
plt.subplot(122)
plt.imshow(dst)
plt.title('Corrected')

## Ex. 7 - Compute reprojection error for all points and write the mean error
# Hints: cv2.projectPoints, cv2.norm
# ---

# ---

# Solution 7
errors = []
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    for j in range(len(img_points[i])):
        point_detected = img_points[i][j]
        point_reprojected = imgpoints2[j]
        error = cv2.norm(point_detected, point_reprojected, cv2.NORM_L2)
        errors.append(error)

np_errors = np.array(errors)
print(f"Mean error: {np_errors.mean()}")
print(f"Median error {np.median(np_errors)}")
plt.hist(np_errors, 20)
plt.show()
