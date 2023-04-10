import numpy as np

"""
Module handles edge detection kernels for the image jobs for the APPL block.
"""


def dilate(kernel, factor):
    x = np.zeros((3 + 2 * factor, (3 + 2 * factor)), dtype=kernel.dtype)

    x[0][0] = kernel[0][0]
    x[0][1 + factor] = kernel[0][1]
    x[0][-1] = kernel[0][2]

    x[1 + factor][0] = kernel[1][0]
    x[1 + factor][1 + factor] = kernel[1][1]
    x[1 + factor][-1] = kernel[1][2]

    x[-1][0] = kernel[2][0]
    x[-1][1 + factor] = kernel[2][1]
    x[-1][-1] = kernel[2][2]

    return x


# Pixel difference operator
pixel_difference_3x3_x = np.array([[0, 0, 0],
                                   [0, -1, +1],
                                   [0, 0, 0]])

pixel_difference_3x3_y = np.array([[0, -1, 0],
                                   [0, +1, 0],
                                   [0, 0, 0]])

assert len(pixel_difference_3x3_x) == len(pixel_difference_3x3_x[0]) == len(pixel_difference_3x3_y) == len(pixel_difference_3x3_y[0]) == 3

# Custom Dilated Sobel Kernels
# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
pixel_difference_dilated_5x5_x = dilate(pixel_difference_3x3_x, 1)
pixel_difference_dilated_5x5_y = dilate(pixel_difference_3x3_y, 1)

assert len(pixel_difference_dilated_5x5_x) == len(pixel_difference_dilated_5x5_x[0]) \
       == len(pixel_difference_dilated_5x5_y) == len(pixel_difference_dilated_5x5_y[0]) == 5

# Custom Dilated Sobel Kernels
# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
pixel_difference_dilated_7x7_x = dilate(pixel_difference_3x3_x, 2)
pixel_difference_dilated_7x7_y = dilate(pixel_difference_3x3_y, 2)

assert len(pixel_difference_dilated_7x7_x) == len(pixel_difference_dilated_7x7_x[0]) \
       == len(pixel_difference_dilated_7x7_y) == len(pixel_difference_dilated_7x7_y[0]) == 7

# Pixel difference operator
separated_pixel_difference_3x3_x = np.array([[0, 0, 0],
                                             [-1, 0, +1],
                                             [0, 0, 0]])

separated_pixel_difference_3x3_y = np.array([[0, -1, 0],
                                             [0, 0, 0],
                                             [0, +1, 0]])

assert len(separated_pixel_difference_3x3_x) == len(separated_pixel_difference_3x3_x[0]) \
       == len(separated_pixel_difference_3x3_y) == len(separated_pixel_difference_3x3_y[0]) == 3

# Custom Dilated Sobel Kernels
# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
separated_pixel_difference_dilated_5x5_x = dilate(separated_pixel_difference_3x3_x, 1)
separated_pixel_difference_dilated_5x5_y = dilate(separated_pixel_difference_3x3_y, 1)

assert len(separated_pixel_difference_dilated_5x5_x) == len(separated_pixel_difference_dilated_5x5_x[0]) \
       == len(separated_pixel_difference_dilated_5x5_y) == len(separated_pixel_difference_dilated_5x5_y[0]) == 5

# Custom Dilated Sobel Kernels
# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
separated_pixel_difference_dilated_7x7_x = dilate(separated_pixel_difference_3x3_x, 2)
separated_pixel_difference_dilated_7x7_y = dilate(separated_pixel_difference_3x3_y, 2)

assert len(separated_pixel_difference_dilated_7x7_x) == len(separated_pixel_difference_dilated_7x7_x[0]) \
       == len(separated_pixel_difference_dilated_7x7_y) == len(separated_pixel_difference_dilated_7x7_y[0]) == 7

# Standard Roberts Kernels
# Roberts, L. (1963).Machine Perception of Three-Dimensional Solids.
roberts_2x2_x = np.array([[1, 0],
                          [0, -1]])
roberts_2x2_y = np.array([[0, 1],
                          [-1, 0]])

assert len(roberts_2x2_x) == len(roberts_2x2_x[0]) == len(roberts_2x2_y) == len(roberts_2x2_y[0]) == 2

# Standard Sobel Kernels
# Sobel, I. and Feldman, G.(1973).  A 3x3 isotropic gradient operator for im-age processing.
sobel_3x3_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

sobel_3x3_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

assert len(sobel_3x3_x) == len(sobel_3x3_x[0]) == len(sobel_3x3_y) == len(sobel_3x3_y[0]) == 3

# Extended Sobel filter 5x5
# Gupta, S. and Mazum-dar, S. G. (2013).  Sobel edge detection algorithm.
# Lateef, R. A. R. (2008). Expansion and implementation of a 3x3 sobel and prewitt edge detection filter to a 5x5 dimension filter
# Levkine,  G. (2012).   Prewitt,  sobel and scharr gradient 5x5 convolution matrices.
# Kekre, H. (2010).  Image segmentation using extended edge operator for mammography images.
sobel_5x5_x = np.array([[-5, -4, 0, 4, 5],
                        [-8, -10, 0, 10, 8],
                        [-10, -20, 0, 20, 10],
                        [-8, -10, 0, 10, 8],
                        [-5, -4, 0, 4, 5]])

sobel_5x5_y = np.array([[-5, -8, -10, -8, -5],
                        [-4, -10, -20, -10, -4],
                        [0, 0, 0, 0, 0],
                        [4, 10, 20, 10, 4],
                        [5, 8, 10, 8, 5]])

assert len(sobel_5x5_x) == len(sobel_5x5_x[0]) == len(sobel_5x5_y) == len(sobel_5x5_y[0]) == 5

# Extended Sobel filter 7x7
# Levkine,  G. (2012).   Prewitt, Sobel and Scharr gradient 5x5 convolution matrices.
sobel_7x7_x = np.array([[-780, -720, -468, 0, 468, 720, 780],
                        [-1080, -1170, -936, 0, 936, 1170, 1080],
                        [-1404, -1872, -2340, 0, 2340, 1872, 1404],
                        [-1560, -2340, -4680, 0, 4680, 2340, 1560],
                        [-1404, -1872, -2340, 0, 2340, 1872, 1404],
                        [-1080, -1170, -936, 0, 936, 1170, 1080],
                        [-780, -720, -468, 0, 468, 720, 780]])

sobel_7x7_y = np.array([[-780, -1080, -1404, -1080, -1404, -1080, -780],
                        [-720, -1170, -1872, -2340, -1872, -1170, -720],
                        [-468, -936, -2340, -4608, -2340, -936, -468],
                        [0, 0, 0, 0, 0, 0, 0],
                        [468, 936, 2340, 4608, 2340, 936, 468],
                        [720, 1170, 1872, 2340, 1872, 1170, 720],
                        [780, 1080, 1404, 1080, 1404, 1080, 780]])

assert len(sobel_7x7_x) == len(sobel_7x7_x[0]) == len(sobel_7x7_y) == len(sobel_7x7_y[0]) == 7

# Dilated Sobel 5x5 Kernels
# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
sobel_dilated_5x5_x = dilate(sobel_3x3_x, 1)
sobel_dilated_5x5_y = dilate(sobel_3x3_y, 1)

assert len(sobel_dilated_5x5_x) == len(sobel_dilated_5x5_x[0]) == len(sobel_dilated_5x5_y) == len(sobel_dilated_5x5_y[0]) == 5

# Custom Dilated Sobel Kernels
# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
sobel_dilated_7x7_x = dilate(sobel_3x3_x, 2)
sobel_dilated_7x7_y = dilate(sobel_3x3_y, 2)

assert len(sobel_dilated_7x7_x) == len(sobel_dilated_7x7_x[0]) == len(sobel_dilated_7x7_y) == len(sobel_dilated_7x7_y[0]) == 7

# Standard Prewitt Kernels
# J. M. S. Prewitt, “Object enhancement and extraction,” Pict. Process. 1970.
prewitt_3x3_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

prewitt_3x3_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])

assert len(prewitt_3x3_x) == len(prewitt_3x3_x[0]) == len(prewitt_3x3_y) == len(prewitt_3x3_y[0]) == 3

# 5x5 Prewitt Kernels  Dr. H. B. Kekre et. al. / (IJCSE) International Journal on Computer Science and Engineering
# 5x5 Prewitt  Levkine,  G. (2012).   Prewitt,  sobel and scharr gradient 5x5 convolution matrices.
# Expansion and Implementation of a 3x3 Sobel and Prewitt Edge Detection Filter to a 5x5 Dimension Filter
prewitt_5x5_x = np.array([[-2, -1, 0, 1, 2],
                          [-2, -1, 0, 1, 2],
                          [-2, -1, 0, 1, 2],
                          [-2, -1, 0, 1, 2],
                          [-2, -1, 0, 1, 2]])

prewitt_5x5_y = np.array([[-2, -2, -2, -2, -2],
                          [-1, -1, -1, -1, -1],
                          [0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1],
                          [2, 2, 2, 2, 2]])

assert len(prewitt_5x5_x) == len(prewitt_5x5_x[0]) == len(prewitt_5x5_y) == len(prewitt_5x5_y[0]) == 5

# 5x5 Prewitt Modified  Levkine,  G. (2012).   Prewitt,  sobel and scharr gradient 5x5 convolution matrices.
prewitt_levkine_5x5_x = np.array([[-2, -1, 0, 1, 2],
                                  [-2, -4, 0, 4, 2],
                                  [-2, -4, 0, 4, 2],
                                  [-2, -4, 0, 4, 2],
                                  [-2, -1, 0, 1, 2]])

prewitt_levkine_5x5_y = np.array([[-2, -2, -2, -2, -2],
                                  [-1, -4, -4, -4, -1],
                                  [0, 0, 0, 0, 0],
                                  [1, 4, 4, 4, 1],
                                  [2, 2, 2, 2, 2]])

assert len(prewitt_levkine_5x5_x) == len(prewitt_levkine_5x5_x[0]) == len(prewitt_levkine_5x5_y) == len(prewitt_levkine_5x5_y[0]) == 5

# 7x7 Prewitt  Levkine,  G. (2012).   Prewitt,  sobel and scharr gradient 5x5 convolution matrices.
prewitt_7x7_x = np.array([[-3, -2, -1, 0, 1, 2, 3],
                          [-3, -2, -1, 0, 1, 2, 3],
                          [-3, -2, -1, 0, 1, 2, 3],
                          [-3, -2, -1, 0, 1, 2, 3],
                          [-3, -2, -1, 0, 1, 2, 3],
                          [-3, -2, -1, 0, 1, 2, 3],
                          [-3, -2, -1, 0, 1, 2, 3]])

prewitt_7x7_y = np.array([[-3, -3, -3, -3, -3, -3, -3],
                          [-2, -2, -2, -2, -2, -2, -2],
                          [-1, -1, -1, -1, -1, -1, -1],
                          [0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1],
                          [2, 2, 2, 2, 2, 2, 2],
                          [3, 3, 3, 3, 3, 3, 3]])

assert len(prewitt_7x7_x) == len(prewitt_7x7_x[0]) == len(prewitt_7x7_y) == len(prewitt_7x7_y[0]) == 7

# 5x5 Prewitt Dilated
# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
prewitt_dilated_5x5_x = dilate(prewitt_3x3_x, 1)
prewitt_dilated_5x5_y = dilate(prewitt_3x3_y, 1)

assert len(prewitt_5x5_x) == len(prewitt_5x5_x[0]) == len(prewitt_5x5_y) == len(prewitt_5x5_y[0]) == 5

# 5x5 Prewitt Dilated
# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
prewitt_dilated_7x7_x = dilate(prewitt_3x3_x, 2)
prewitt_dilated_7x7_y = dilate(prewitt_3x3_y, 2)

assert len(prewitt_7x7_x) == len(prewitt_7x7_x[0]) == len(prewitt_7x7_y) == len(prewitt_7x7_y[0]) == 7

# Scharr standard filter
# Scharr, H. (2000). Optimal operators in digital image processing.
scharr_3x3_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]])
scharr_3x3_y = np.array([[-3, -10, -3],
                         [0, 0, 0],
                         [3, 10, 3]])

assert len(scharr_3x3_x) == len(scharr_3x3_x[0]) == len(scharr_3x3_y) == len(scharr_3x3_y[0]) == 3

# Scharr 5x5 filter
# Levkine,  G. (2012).   Prewitt,  sobel and scharr gradient 5x5 convolution matrices.
scharr_5x5_x = np.array([[-1, -1, 0, 1, 1],
                         [-2, -2, 0, 2, 2],
                         [-3, -6, 0, 6, 3],
                         [-2, -2, 0, 2, 2],
                         [-1, -1, 0, 1, 1]])

scharr_5x5_y = np.array([[-1, -2, -3, -2, -1],
                         [-1, -2, -6, -2, -1],
                         [0, 0, 0, 0, 0],
                         [1, 2, 6, 2, 1],
                         [1, 2, 3, 2, 1]])

assert len(scharr_5x5_x) == len(scharr_5x5_x[0]) == len(scharr_5x5_y) == len(scharr_5x5_y[0]) == 5

# Scharr 5x5 filter
# Chen, D. Chen, S. Meng, A novel region selection algorithm for auto-focusing method based on depth from focus
scharr_chen_5x5_x = np.array([[-2, -3, 0, 3, 2],
                              [-3, -4, 0, 4, 3],
                              [-6, -6, 0, 6, 6],
                              [-3, -4, 0, 4, 3],
                              [-2, -3, 0, 3, 2]])

scharr_chen_5x5_y = np.array([[-2, -3, -6, -3, -2],
                              [-3, -4, -6, -4, -3],
                              [0, 0, 0, 0, 0],
                              [3, 4, 6, 4, 3],
                              [2, 3, 6, 3, 2]])

assert len(scharr_chen_5x5_x) == len(scharr_chen_5x5_x[0]) == len(scharr_chen_5x5_y) == len(scharr_chen_5x5_y[0]) == 5

# Scharr dilated 5x5 filter
# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
scharr_dilated_5x5_x = dilate(scharr_3x3_x, 1)
scharr_dilated_5x5_y = dilate(scharr_3x3_y, 1)

assert len(scharr_dilated_5x5_x) == len(scharr_dilated_5x5_x[0]) == len(scharr_dilated_5x5_y) == len(scharr_dilated_5x5_y[0]) == 5

scharr_dilated_7x7_x = dilate(scharr_3x3_x, 2)
scharr_dilated_7x7_y = dilate(scharr_3x3_y, 2)

assert len(scharr_dilated_7x7_x) == len(scharr_dilated_7x7_x[0]) == len(scharr_dilated_7x7_y) == len(scharr_dilated_7x7_y[0]) == 7

# Kirch standard filter
# R. A. Kirsch, Computer determination of the constituent structure of biological images, Computers and biomedical research 4 (1971) 315–328.
kirsch_3x3_x = np.array([[5, 5, 5],
                         [-3, 0, -3],
                         [-3, -3, -3]])

kirsch_3x3_y = np.array([[5, -3, -3],
                         [5, 0, -3],
                         [5, -3, -3]])

assert len(kirsch_3x3_x) == len(kirsch_3x3_x[0]) == len(kirsch_3x3_y) == len(kirsch_3x3_y[0]) == 3

# Kirch 5x5 filter standard
# Kekre, H. (2010).  Image segmentation using extended edge operator for mammography images.
kirsch_5x5_x = np.array([[-7, -7, -7, 9, 9],
                         [-7, -3, -3, 5, 9],
                         [-7, -3, 0, 5, 9],
                         [-7, -3, -3, 5, 9],
                         [-7, -7, -7, 9, 9]])

kirsch_5x5_y = np.array([[-7, -7, -7, -7, -7],
                         [-7, -3, -3, -3, -7],
                         [-7, -3, 0, -3, -7],
                         [9, 5, 5, 5, 9],
                         [9, 9, 9, 9, 9]])

assert len(kirsch_5x5_x) == len(kirsch_5x5_x[0]) == len(kirsch_5x5_y) == len(kirsch_5x5_y[0]) == 5

# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
kirsch_dilated_5x5_x = dilate(kirsch_3x3_x, 1)
kirsch_dilated_5x5_y = dilate(kirsch_3x3_y, 1)

assert len(kirsch_dilated_5x5_x) == len(kirsch_dilated_5x5_x[0]) == len(kirsch_dilated_5x5_y) == len(kirsch_dilated_5x5_y[0]) == 5

# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
kirsch_dilated_7x7_x = dilate(kirsch_3x3_x, 2)
kirsch_dilated_7x7_y = dilate(kirsch_3x3_y, 2)

assert len(kirsch_dilated_7x7_x) == len(kirsch_dilated_7x7_x[0]) == len(kirsch_dilated_7x7_y) == len(kirsch_dilated_7x7_y[0]) == 7

# Orhei-Vert-Vasiu 3x3 Operator
# C. Orhei, S. Vert, R. Vasiu, A novel edge detection operator for identifying buildings in augmented reality applications (2020).
orhei_3x3_x = np.array([[-1, 0, 1],
                        [-4, 0, 4],
                        [-1, 0, 1]])

orhei_3x3_y = np.array([[-1, -4, -1],
                        [0, 0, 0],
                        [1, 4, 1]])

assert len(orhei_3x3_x) == len(orhei_3x3_x[0]) == len(orhei_3x3_y) == len(orhei_3x3_y[0]) == 3

# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
orhei_dilated_5x5_x = dilate(orhei_3x3_x, 1)
orhei_dilated_5x5_y = dilate(orhei_3x3_y, 1)

assert len(orhei_dilated_5x5_x) == len(orhei_dilated_5x5_x[0]) == len(orhei_dilated_5x5_y) == len(orhei_dilated_5x5_y[0]) == 5

orhei_dilated_7x7_x = dilate(orhei_3x3_x, 2)
orhei_dilated_7x7_y = dilate(orhei_3x3_y, 2)
assert len(orhei_dilated_7x7_x) == len(orhei_dilated_7x7_x[0]) == len(orhei_dilated_7x7_y) == len(orhei_dilated_7x7_y[0]) == 7

orhei_dilated_9x9_x = dilate(orhei_3x3_x, 3)
orhei_dilated_9x9_y = dilate(orhei_3x3_y, 3)
assert len(orhei_dilated_9x9_x) == len(orhei_dilated_9x9_x[0]) == len(orhei_dilated_9x9_y) == len(orhei_dilated_9x9_y[0]) == 9

# Orhei-Vert-Vasiu 5x5 Operator
# C. Orhei, S. Vert, R. Vasiu, A novel edge detection operator for identifying buildings in augmented reality applications (2020).
orhei_v1_5x5_x = np.array([[-25, -4, 0, 4, 25],
                           [-64, -10, 0, 10, 64],
                           [-100, -20, 0, 20, 100],
                           [-64, -10, 0, 10, 64],
                           [-25, -4, 0, 4, 25]])

orhei_v1_5x5_y = np.array([[-25, -4, -100, -4, -25],
                           [-64, -10, -20, -10, -64],
                           [0, 0, 0, 0, 0],
                           [64, 10, 20, 10, 64],
                           [25, 4, 100, 4, 25]])

assert len(orhei_v1_5x5_x) == len(orhei_v1_5x5_x[0]) == len(orhei_v1_5x5_y) == len(orhei_v1_5x5_y[0]) == 5

# Orhei-Vert-Vasiu 5x5 Operator
# C. Orhei, S. Vert, R. Vasiu, A novel edge detection operator for identifying buildings in augmented reality applications (2020).
orhei_5x5_x = np.array([[-2, -1, 0, 1, 2],
                        [-2, -1, 0, 1, 2],
                        [-8, -4, 0, 4, 8],
                        [-2, -1, 0, 1, 2],
                        [-2, -1, 0, 1, 2]])

orhei_5x5_y = np.array([[-2, -2, -8, -2, -2],
                        [-1, -1, -4, -1, -1],
                        [0, 0, 0, 0, 0],
                        [1, 1, 4, 1, 1],
                        [2, 2, 8, 2, 2]])

assert len(orhei_5x5_x) == len(orhei_5x5_x[0]) == len(orhei_5x5_y) == len(orhei_5x5_y[0]) == 5

# Frei-Chen operator
# the masks are normalised
# Frei and Chung-Ching Chen, “Fast Boundary Detection: A Generalization and a New Algorithm,” IEEE Trans. , vol. C–26, no. 10, pp. 988–998, Oct. 1977, doi: 10.1109/TC.1977.1674733.
frei_chen_v1 = np.array([[-0.3535533845424652, 0, 0.3535533845424652],
                         [-0.5, 0, 0.5],
                         [-0.3535533845424652, 0, 0.3535533845424652]])

frei_chen_v2 = np.array([[-0.3535533845424652, -0.5, -0.3535533845424652],
                         [0, 0, 0],
                         [0.3535533845424652, 0.5, 0.3535533845424652]])

frei_chen_v3 = np.array([[0, -0.3535533845424652, 0.5],
                         [0.3535533845424652, 0, -0.3535533845424652],
                         [-0.5, 0.3535533845424652, 0]])

frei_chen_v4 = np.array([[0.5, -0.3535533845424652, 0],
                         [-0.3535533845424652, 0, 0.3535533845424652],
                         [0, 0.3535533845424652, -0.5]])

frei_chen_v5 = np.array([[0, 0.5, 0],
                         [-0.5, 0, -0.5],
                         [0, 0.5, 0]])

frei_chen_v6 = np.array([[-0.5, 0, 0.5],
                         [0, 0, 0],
                         [0.5, 0, -0.5]])

frei_chen_v7 = np.array([[0.1666666716337204, -0.3333333432674408, 0.1666666716337204],
                         [-0.3333333432674408, 0.6666666865348816, -0.3333333432674408],
                         [0.1666666716337204, -0.3333333432674408, 0.1666666716337204]])

frei_chen_v8 = np.array([[-0.3333333432674408, 0.1666666716337204, -0.3333333432674408],
                         [0.1666666716337204, 0.6666666865348816, 0.1666666716337204],
                         [-0.3333333432674408, 0.1666666716337204, -0.3333333432674408]])

frei_chen_v9 = np.array([[0.3333333432674408, 0.3333333432674408, 0.3333333432674408],
                         [0.3333333432674408, 0.3333333432674408, 0.3333333432674408],
                         [0.3333333432674408, 0.3333333432674408, 0.3333333432674408]])

frei_chen_3x3_x = frei_chen_v1
frei_chen_3x3_y = frei_chen_v2

assert len(frei_chen_v1) == len(frei_chen_v1[0]) == len(frei_chen_v2) == len(frei_chen_v2[0]) == \
       len(frei_chen_v3) == len(frei_chen_v3[0]) == len(frei_chen_v4) == len(frei_chen_v4[0]) == \
       len(frei_chen_v5) == len(frei_chen_v5[0]) == len(frei_chen_v6) == len(frei_chen_v6[0]) == \
       len(frei_chen_v7) == len(frei_chen_v7[0]) == len(frei_chen_v8) == len(frei_chen_v8[0]) == \
       len(frei_chen_v9) == len(frei_chen_v9[0]) == 3

# Frei-Chen Dilated
frei_chen_dilated_5x5_x = dilate(frei_chen_3x3_x, 1)
frei_chen_dilated_5x5_y = dilate(frei_chen_3x3_y, 1)

assert len(frei_chen_dilated_5x5_x) == len(frei_chen_dilated_5x5_x[0]) == len(frei_chen_dilated_5x5_y) == len(
    frei_chen_dilated_5x5_y[0]) == 5

frei_chen_dilated_7x7_x = dilate(frei_chen_3x3_x, 2)
frei_chen_dilated_7x7_y = dilate(frei_chen_3x3_y, 2)
assert len(frei_chen_dilated_7x7_x) == len(frei_chen_dilated_7x7_x[0]) == len(frei_chen_dilated_7x7_y) == len(
    frei_chen_dilated_7x7_y[0]) == 7

# Frei-Chen Dilated
frei_chen_dilated_5x5_v1 = dilate(kernel=frei_chen_v1, factor=1)
frei_chen_dilated_5x5_v2 = dilate(kernel=frei_chen_v2, factor=1)
frei_chen_dilated_5x5_v3 = dilate(kernel=frei_chen_v3, factor=1)
frei_chen_dilated_5x5_v4 = dilate(kernel=frei_chen_v4, factor=1)
frei_chen_dilated_5x5_v5 = dilate(kernel=frei_chen_v5, factor=1)
frei_chen_dilated_5x5_v6 = dilate(kernel=frei_chen_v6, factor=1)
frei_chen_dilated_5x5_v7 = dilate(kernel=frei_chen_v7, factor=1)
frei_chen_dilated_5x5_v8 = dilate(kernel=frei_chen_v8, factor=1)
frei_chen_dilated_5x5_v9 = dilate(kernel=frei_chen_v9, factor=1)

assert len(frei_chen_dilated_5x5_v1) == len(frei_chen_dilated_5x5_v1[0]) == len(frei_chen_dilated_5x5_v2) == len(frei_chen_dilated_5x5_v2[0]) == \
       len(frei_chen_dilated_5x5_v3) == len(frei_chen_dilated_5x5_v3[0]) == len(frei_chen_dilated_5x5_v4) == len(frei_chen_dilated_5x5_v4[0]) == \
       len(frei_chen_dilated_5x5_v5) == len(frei_chen_dilated_5x5_v5[0]) == len(frei_chen_dilated_5x5_v6) == len(frei_chen_dilated_5x5_v6[0]) == \
       len(frei_chen_dilated_5x5_v7) == len(frei_chen_dilated_5x5_v7[0]) == len(frei_chen_dilated_5x5_v8) == len(frei_chen_dilated_5x5_v8[0]) == \
       len(frei_chen_dilated_5x5_v9) == len(frei_chen_dilated_5x5_v9[0]) == 5

# Frei-Chen Dilated

frei_chen_dilated_7x7_v1 = dilate(kernel=frei_chen_v1, factor=2)
frei_chen_dilated_7x7_v2 = dilate(kernel=frei_chen_v2, factor=2)
frei_chen_dilated_7x7_v3 = dilate(kernel=frei_chen_v3, factor=2)
frei_chen_dilated_7x7_v4 = dilate(kernel=frei_chen_v4, factor=2)
frei_chen_dilated_7x7_v5 = dilate(kernel=frei_chen_v5, factor=2)
frei_chen_dilated_7x7_v6 = dilate(kernel=frei_chen_v6, factor=2)
frei_chen_dilated_7x7_v7 = dilate(kernel=frei_chen_v7, factor=2)
frei_chen_dilated_7x7_v8 = dilate(kernel=frei_chen_v8, factor=2)
frei_chen_dilated_7x7_v9 = dilate(kernel=frei_chen_v9, factor=2)

assert len(frei_chen_dilated_7x7_v1) == len(frei_chen_dilated_7x7_v1[0]) == len(frei_chen_dilated_7x7_v2) == len(frei_chen_dilated_7x7_v2[0]) == \
       len(frei_chen_dilated_7x7_v3) == len(frei_chen_dilated_7x7_v3[0]) == len(frei_chen_dilated_7x7_v4) == len(frei_chen_dilated_7x7_v4[0]) == \
       len(frei_chen_dilated_7x7_v5) == len(frei_chen_dilated_7x7_v5[0]) == len(frei_chen_dilated_7x7_v6) == len(frei_chen_dilated_7x7_v6[0]) == \
       len(frei_chen_dilated_7x7_v7) == len(frei_chen_dilated_7x7_v7[0]) == len(frei_chen_dilated_7x7_v8) == len(frei_chen_dilated_7x7_v8[0]) == \
       len(frei_chen_dilated_7x7_v9) == len(frei_chen_dilated_7x7_v9[0]) == 7



# Kayyali Operator
# E. Kawalec-Latała, Edge detection on images of pseudo impedance section supported by context and adaptive transformation model images(2014)
kayyali_3x3_x = np.array([[6, 0, -6],
                          [0, 0, 0],
                          [-6, 0, 6]])

kayyali_3x3_y = np.array([[-6, 0, 6],
                          [0, 0, 0],
                          [6, 0, -6]])

assert len(kayyali_3x3_x) == len(kayyali_3x3_x[0]) == len(kayyali_3x3_y) == len(kayyali_3x3_y[0]) == 3

# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
kayyali_dilated_5x5_x = dilate(kayyali_3x3_x, 1)
kayyali_dilated_5x5_y = dilate(kayyali_3x3_y, 1)

assert len(kayyali_dilated_5x5_x) == len(kayyali_dilated_5x5_x[0]) == len(kayyali_dilated_5x5_y) == len(kayyali_dilated_5x5_y[0]) == 5

# Bogdan, V., Bonchis, C., & Orhei, C. (2020). Custom dilated edge detection filters.
kayyali_dilated_7x7_x = dilate(kayyali_3x3_x, 2)
kayyali_dilated_7x7_y = dilate(kayyali_3x3_y, 2)

assert len(kayyali_dilated_7x7_x) == len(kayyali_dilated_7x7_x[0]) == len(kayyali_dilated_7x7_y) == len(kayyali_dilated_7x7_y[0]) == 7

# El-Arwadi and El-Zaart  Operator
# T. El-Arwadi, A. El-Zaart, A novel 5x5 edge detection operator for blood vessel images, Current Journal of AppliedScience and Technology (2015).
el_arwadi_el_zaart_5x5_x = np.array([[1, 0, 0, 0, 1],
                                     [0, 1, 0, 1, 0],
                                     [0, 0, -12, 0, 0],
                                     [0, 2, 0, 2, 0],
                                     [2, 0, 0, 0, 2]])

el_arwadi_el_zaart_5x5_y = np.array([[1, 0, 0, 0, 2],
                                     [0, 1, 0, 2, 0],
                                     [0, 0, -12, 0, 0],
                                     [0, 1, 0, 2, 0],
                                     [1, 0, 0, 0, 2]])

assert len(el_arwadi_el_zaart_5x5_x) == len(el_arwadi_el_zaart_5x5_x[0]) == len(el_arwadi_el_zaart_5x5_y) == len(
    el_arwadi_el_zaart_5x5_y[0]) == 5

# Prewitt Compass Operator
# J. M. Prewitt, Object enhancement and extraction, Picture processing and Psychopictorics 10 (1970) 15–19.
prewitt_compass_3x3_x = np.array([[1, 1, -1],
                                  [1, -2, -1],
                                  [1, 1, -1]])

assert len(prewitt_compass_3x3_x) == len(prewitt_compass_3x3_x[0]) == 3

prewitt_compass_dilated_5x5_x = dilate(prewitt_compass_3x3_x, 1)

assert len(prewitt_compass_dilated_5x5_x) == len(prewitt_compass_dilated_5x5_x[0]) == 5

prewitt_compass_dilated_7x7_x = dilate(prewitt_compass_3x3_x, 2)

assert len(prewitt_compass_dilated_7x7_x) == len(prewitt_compass_dilated_7x7_x[0]) == 7

# Navatia Babu  Operator
# R. Nevatia, K. R. Babu,  Linear feature extraction and description,  Computer Graphics and Image Processing 13(1980) 257–269.

# 0 degree
navatia_babu_5x5_g4 = np.array([[-100, -100, 0, 100, 100],
                                [-100, -100, 0, 100, 100],
                                [-100, -100, 0, 100, 100],
                                [-100, -100, 0, 100, 100],
                                [-100, -100, 0, 100, 100]])
# 30 degree
navatia_babu_5x5_g2 = np.array([[-100, 32, 100, 100, 100],
                                [-100, -78, 92, 100, 100],
                                [-100, -100, 0, 100, 100],
                                [-100, -100, -92, 78, 100],
                                [-100, -100, -100, -32, 100]])
# 60 degree
navatia_babu_5x5_g3 = np.array([[100, 100, 100, 100, 100],
                                [-32, 78, 100, 100, 100],
                                [-100, -92, 0, 92, 100],
                                [-100, -100, -100, -78, 32],
                                [-100, -100, -100, -100, -100]])

# 90 degree
navatia_babu_5x5_g1 = np.array([[100, 100, 100, 100, 100],
                                [100, 100, 100, 100, 100],
                                [0, 0, 0, 0, 0],
                                [-100, -100, -100, -100, -100],
                                [-100, -100, -100, -100, -100]])
# 120 degree
navatia_babu_5x5_g5 = np.array([[-100, 100, 100, 100, 100],
                                [-100, 100, 100, 78, -32],
                                [-100, 92, 0, -92, -100],
                                [32, -78, -100, -100, -100],
                                [-100, -100, -100, -100, -100]])
# 150 degree
navatia_babu_5x5_g6 = np.array([[100, 100, 100, 32, -100],
                                [100, 100, 92, -78, -100],
                                [100, 100, 0, -100, -100],
                                [100, 78, -92, -100, -100],
                                [100, -32, -100, -100, -100]])

assert len(navatia_babu_5x5_g1) == len(navatia_babu_5x5_g1[0]) == len(navatia_babu_5x5_g2) == len(navatia_babu_5x5_g2[0]) \
       == len(navatia_babu_5x5_g3) == len(navatia_babu_5x5_g3[0]) == len(navatia_babu_5x5_g4) == len(navatia_babu_5x5_g4[0]) \
       == len(navatia_babu_5x5_g5) == len(navatia_babu_5x5_g5[0]) == len(navatia_babu_5x5_g6) == len(navatia_babu_5x5_g6[0]) \
       == 5

# Kroon Operator
# D. Kroon, Numerical optimization of kernel based image derivatives, Short Paper University Twente (2009).
kroon_3x3_x = np.array([[-17, 0, 17],
                        [-61, 0, 61],
                        [-17, 0, 17]])

kroon_3x3_y = np.array([[-17, -61, -17],
                        [0, 0, 0],
                        [17, 61, 17]])

assert len(kroon_3x3_x) == len(kroon_3x3_x[0]) == len(kroon_3x3_y) == len(kroon_3x3_y[0]) == 3

# Kroon Operator Dilated 5x5
kroon_dilated_5x5_x = dilate(kroon_3x3_x, 1)
kroon_dilated_5x5_y = dilate(kroon_3x3_y, 1)

assert len(kroon_dilated_5x5_x) == len(kroon_dilated_5x5_x[0]) == len(kroon_dilated_5x5_y) == len(kroon_dilated_5x5_y[0]) == 5

# Kroon Operator Dilated 7x7
kroon_dilated_7x7_x = dilate(kroon_3x3_x, 2)
kroon_dilated_7x7_y = dilate(kroon_3x3_y, 2)

assert len(kroon_dilated_7x7_x) == len(kroon_dilated_7x7_x[0]) == len(kroon_dilated_7x7_y) == len(kroon_dilated_7x7_y[0]) == 7

# Kitchen-Malin
# Kitchen, J. Malin,  The effect of spatial discretization on the magnitude and direction response of simple differential edge operators on a step edge
kitchen_3x3_x = np.array([[-2, 0, 2],
                          [-3, 0, 3],
                          [-2, 0, 3]])

kitchen_3x3_y = np.array([[-2, -3, -2],
                          [0, 0, 0],
                          [2, 3, 2]])

assert len(kitchen_3x3_x) == len(kitchen_3x3_x[0]) == len(kitchen_3x3_y) == len(kitchen_3x3_y[0]) == 3

# Kitchen-Malin Operator Dilated
kitchen_dilated_5x5_x = dilate(kitchen_3x3_x, 1)
kitchen_dilated_5x5_y = dilate(kitchen_3x3_y, 1)

assert len(kitchen_dilated_5x5_x) == len(kitchen_dilated_5x5_x[0]) == len(kitchen_dilated_5x5_y) == len(kitchen_dilated_5x5_y[0]) == 5

kitchen_dilated_7x7_x = dilate(kitchen_3x3_x, 2)
kitchen_dilated_7x7_y = dilate(kitchen_3x3_y, 2)

assert len(kitchen_dilated_7x7_x) == len(kitchen_dilated_7x7_x[0]) == len(kitchen_dilated_7x7_y) == len(kitchen_dilated_7x7_y[0]) == 7

# Pyramid Operator
pyramid_7x7_x = np.array([[-1, -1, -1, 0, 1, 1, 1],
                          [-1, -2, -2, 0, 2, 2, 1],
                          [-1, -2, -3, 0, 3, 2, 1],
                          [-1, -2, -3, 0, 3, 2, 1],
                          [-1, -2, -3, 0, 3, 2, 1],
                          [-1, -2, -2, 0, 2, 2, 1],
                          [-1, -1, -1, 0, 1, 1, 1]])

pyramid_7x7_y = np.array([[-1, -1, -1, -1, -1, -1, -1],
                          [-1, -2, -2, -2, -2, -2, -1],
                          [-1, -2, -3, -3, -3, -2, -1],
                          [0, 0, 0, 0, 0, 0, 0],
                          [1, 2, 3, 3, 3, 2, 1],
                          [1, 2, 2, 2, 2, 2, 1],
                          [1, 1, 1, 1, 1, 1, 1]])

# https://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter5.pdf
laplace_v1_3x3_xy = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

laplace_v1_5x5_xy = np.array([[0, 0,   1, 0, 0],
                              [0, 1,   2, 1, 0],
                              [1, 2, -16, 2, 1],
                              [0, 1,   2, 1, 0],
                              [0, 0,   1, 0, 0]])

laplace_v1_7x7_xy = np.array([[ 0, 0, 1, 1, 1, 0, 0],
                              [ 0, 1, 3, 3, 3, 1, 0],
                              [ 1, 3, 0, -7, 0, 3, 1],
                              [ 1, 3, -7, -24, -7, 3, 1],
                              [ 1, 3, 0, -7, 0, 3, 1],
                              [ 0, 1, 3, 3, 3, 1, 0],
                              [ 0, 0, 1, 1, 1, 0, 0]])

laplace_v1_9x9_xy = np.array( [[0, 0,  1,   1,   1,   1,   1, 0, 0],
                               [0, 1,  3,   3,   3,   3,   3, 1, 0],
                               [1, 3,  7,   7,   7,   7,   7, 3, 1],
                               [1, 3,  7,  -3, -23,  -3,   7, 3, 1],
                               [1, 3,  7, -23, -92, -23,   7, 3, 1],
                               [1, 3,  7,  -3, -23,  -3,   7, 3, 1],
                               [1, 3,  7,   7,   7,   7,   7, 3, 1],
                               [0, 1,  3,   3,   3,   3,   3, 1, 0],
                               [0, 0,  1,   1,   1,   1,   1, 0, 0]])

laplace_v1_dilated_5x5_xy = dilate(laplace_v1_3x3_xy, 1)
laplace_v1_dilated_7x7_xy = dilate(laplace_v1_3x3_xy, 2)
laplace_v1_dilated_9x9_xy = dilate(laplace_v1_3x3_xy, 3)
laplace_v1_dilated_11x11_xy = dilate(laplace_v1_3x3_xy, 4)

# http://fourier.eng.hmc.edu/e161/lectures/gradient/node7.html
laplace_v2_3x3_xy = np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]])

laplace_v2_5x5_xy = np.array([[1, 1,   1, 1, 1],
                              [1, 1,   1, 1, 1],
                              [1, 1, -24, 1, 1],
                              [1, 1,   1, 1, 1],
                              [1, 1,   1, 1, 1]])

laplace_v2_dilated_5x5_xy = dilate(laplace_v2_3x3_xy, 1)
laplace_v2_dilated_7x7_xy = dilate(laplace_v2_3x3_xy, 2)

# https://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter5.pdf
laplace_v3_3x3_xy = np.array([[1, 4, 1],
                              [4, -20, 4],
                              [1, 4, 1]])

laplace_v3_dilated_5x5_xy = dilate(laplace_v3_3x3_xy, 1)
laplace_v3_dilated_7x7_xy = dilate(laplace_v3_3x3_xy, 2)

laplace_v4_3x3_xy = np.array([[-1, 2, -1],
                              [2, -4, 2],
                              [-1, 2, -1]])

laplace_v4_dilated_5x5_xy = dilate(laplace_v4_3x3_xy, 1)
laplace_v4_dilated_7x7_xy = dilate(laplace_v4_3x3_xy, 2)

# https://sci-hub.tw/https://www.sciencedirect.com/science/article/abs/pii/016516849290051W
laplace_v5_3x3_xy = np.array([[2, -1, 2],
                              [-1, -4, -1],
                              [2, -1, 2]])

laplace_v5_dilated_5x5_xy = dilate(laplace_v5_3x3_xy, 1)
laplace_v5_dilated_7x7_xy = dilate(laplace_v5_3x3_xy, 2)

# kernels for LoG precalculated
log_v1_5x5_xy = np.array([[0, 0, 1, 0, 0],
                          [0, 1, 2, 1, 0],
                          [1, 2, -16, 2, 1],
                          [0, 1, 2, 1, 0],
                          [0, 0, 1, 0, 0]])

log_v1_7x7_xy = np.array([[0, 0, 1,    1,  1, 0, 0],
                          [0, 1, 3,    3,  3, 1, 0],
                          [1, 3, 0,   -7,  0, 3, 1],
                          [1, 3, -7, -24, -7, 3, 1],
                          [1, 3, 0,   -7,  0, 3, 1],
                          [0, 1, 3,    3,  3, 1, 0],
                          [0, 0, 1,    1,  1, 0, 0]])

log_v1_9x9_xy = np.array([[0, 0, 3,   2,   2,   2, 3, 0, 0],
                          [0, 2, 3,   5,   5,   5, 3, 2, 0],
                          [3, 3, 5,   3,   0,   3, 5, 3, 3],
                          [2, 5, 3, -12, -23, -12, 3, 5, 2],
                          [2, 5, 0, -23, -40, -23, 0, 5, 2],
                          [2, 5, 3, -12, -23, -12, 3, 5, 2],
                          [3, 3, 5,   3,   0,   3, 5, 3, 3],
                          [0, 2, 3,   5,   5,   5, 3, 2, 0],
                          [0, 0, 3,   2,   2,   2, 3, 0, 0]])


if __name__ == "__main__":
    pass
