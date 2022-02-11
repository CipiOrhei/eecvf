import cv2
import numpy as np

"""
Module handles misc functionalities of the APPL block
"""


def rotate_around_center(mat: np.array) -> np.array:
    """
    Rotate matrix
    :param mat: matrix to apply rotation on
    :return: rotated matrix
    """
    if mat.shape == (3, 3):
        return rotate_matrix_3x3(mat)
    elif mat.shape == (5, 5):
        return rotate_matrix_5x5(mat)
    elif mat.shape == (7, 7):
        return rotate_matrix_7x7(mat)


def rotate_matrix_3x3(mat):
    """
    Rotating around center of matrix the elements
    :param mat: np array matrix to rotate
    :return: new matrix
    """
    t = np.zeros(mat.shape, mat.dtype)
    t[0][0] = mat[1][0]
    t[0][1] = mat[0][0]
    t[0][2] = mat[0][1]
    t[1][0] = mat[2][0]
    t[1][1] = mat[1][1]
    t[1][2] = mat[0][2]
    t[2][0] = mat[2][1]
    t[2][1] = mat[2][2]
    t[2][2] = mat[1][2]

    return t


def rotate_matrix_5x5(mat):
    """
    Rotating around center of matrix the elements
    :param mat: np array matrix to rotate
    :return: new matrix
    """
    t = np.zeros(mat.shape, mat.dtype)
    t[0][2] = mat[0][0]
    t[0][3] = mat[0][1]
    t[0][4] = mat[0][2]
    t[1][4] = mat[0][3]
    t[2][4] = mat[0][4]

    t[0][1] = mat[1][0]
    t[1][2] = mat[1][1]
    t[1][3] = mat[1][2]
    t[2][3] = mat[1][3]
    t[3][4] = mat[1][4]

    t[0][0] = mat[2][0]
    t[1][1] = mat[2][1]
    t[2][2] = mat[2][2]
    t[3][3] = mat[2][3]
    t[4][4] = mat[2][4]

    t[1][0] = mat[3][0]
    t[2][1] = mat[3][1]
    t[3][1] = mat[3][2]
    t[3][2] = mat[3][3]
    t[4][3] = mat[3][4]

    t[2][0] = mat[4][0]
    t[3][0] = mat[4][1]
    t[4][0] = mat[4][2]
    t[4][1] = mat[4][3]
    t[4][2] = mat[4][4]

    return t


def rotate_matrix_7x7(mat):
    """
    Rotating around center of matrix the elements
    :param mat: np array matrix to rotate
    :return: new matrix
    """
    t = np.zeros(mat.shape, mat.dtype)
    t[0][3] = mat[0][0]
    t[0][4] = mat[0][1]
    t[0][5] = mat[0][2]
    t[0][6] = mat[0][3]
    t[1][6] = mat[0][4]
    t[2][6] = mat[0][5]
    t[3][6] = mat[0][6]

    t[0][2] = mat[1][0]
    t[1][3] = mat[1][1]
    t[1][4] = mat[1][2]
    t[1][5] = mat[1][3]
    t[2][5] = mat[1][4]
    t[3][5] = mat[1][5]
    t[4][6] = mat[1][6]

    t[1][0] = mat[2][0]
    t[1][2] = mat[2][1]
    t[2][3] = mat[2][2]
    t[2][4] = mat[2][3]
    t[3][4] = mat[2][4]
    t[4][5] = mat[2][5]
    t[5][6] = mat[2][6]

    t[0][0] = mat[3][0]
    t[1][1] = mat[3][1]
    t[2][2] = mat[3][2]
    t[3][3] = mat[3][3]
    t[4][4] = mat[3][4]
    t[5][5] = mat[3][5]
    t[6][6] = mat[3][6]

    t[1][0] = mat[4][0]
    t[2][1] = mat[4][1]
    t[3][2] = mat[4][2]
    t[4][2] = mat[4][3]
    t[4][3] = mat[4][4]
    t[5][4] = mat[4][5]
    t[6][5] = mat[4][6]

    t[2][0] = mat[5][0]
    t[3][1] = mat[5][1]
    t[4][1] = mat[5][2]
    t[5][1] = mat[5][3]
    t[5][2] = mat[5][4]
    t[5][3] = mat[5][5]
    t[6][4] = mat[5][6]

    t[3][0] = mat[6][0]
    t[4][0] = mat[6][1]
    t[5][0] = mat[6][2]
    t[6][0] = mat[6][3]
    t[6][1] = mat[6][4]
    t[6][2] = mat[6][5]
    t[6][3] = mat[6][6]

    return t


def save_keypoint_to_array(kp: cv2.KeyPoint):
    """
    Function to transfer data from cv2.KeyPoint to an array.
    :param mat: cv2.KeyPoint object
    :return: array of data from keypoints
    """
    tmp_array = np.zeros((7))

    tmp_array[0] = kp.angle
    tmp_array[1] = kp.class_id
    tmp_array[2] = kp.octave
    tmp_array[3] = kp.pt[0]
    tmp_array[4] = kp.pt[1]
    tmp_array[5] = kp.response
    tmp_array[6] = kp.size

    return tmp_array


def return_kp_cv2_object(kp_array):
    """
    Function to transfer data from array to cv2.KeyPoint.
    :param array of data from key-points
    :return: mat: cv2.KeyPoint object
    """
    kp = cv2.KeyPoint(x=kp_array[3], y=kp_array[4], _size=kp_array[6], _angle=kp_array[0], _response=kp_array[5], _octave=int(kp_array[2]), _class_id=int(kp_array[1]))

    return kp


if __name__ == "__main__":
    pass
