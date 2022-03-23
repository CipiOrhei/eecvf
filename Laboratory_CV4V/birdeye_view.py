##
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_H = 375
IMAGE_W = 1242
OUT_H_FACTOR = 8
src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [848, 189], [927, 189]])
dst = np.float32([[500, IMAGE_H * OUT_H_FACTOR], [IMAGE_W - 500, IMAGE_H * OUT_H_FACTOR], [500, 0], [IMAGE_W - 500, 0]])
M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation

##
img = cv2.imread('test_image.png')  # Read the test img
plt.imshow(img)  # Show results
img = img[400:(400+IMAGE_H), 0:IMAGE_W]  # Apply np slicing for ROI crop
warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))  # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))  # Show results
plt.show()
