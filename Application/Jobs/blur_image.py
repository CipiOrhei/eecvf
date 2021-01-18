# noinspection PyPackageRequirements
import cv2
import numpy as np

from numba import jit

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave, Port
from PIL import ImageFilter, Image

from Utils.log_handler import log_error_to_console

"""
Module handles the blurring and smoothing image processing jobs for the APPL block
"""


def init_func() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def main_gaussian_blur_func(port_list: list = None) -> bool:
    """
    Main function for blur job. Done using cv2 library.

    :param port_list: Param needed list of port names [input, wave_offset, kernel_size, sigma, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # check if param OK

    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_POS = 2
    # noinspection PyPep8Naming
    PORT_SIGMA_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_POS = 4

    if len(port_list) != 5:
        log_error_to_console("GAUSSIAN BLUR JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                if port_list[PORT_KERNEL_POS] != 0 or port_list[PORT_SIGMA_POS]:
                    p_out.arr[:] = cv2.GaussianBlur(src=p_in.arr.copy(),
                                                    ksize=(port_list[PORT_KERNEL_POS], port_list[PORT_KERNEL_POS]),
                                                    sigmaX=port_list[PORT_SIGMA_POS],
                                                    sigmaY=port_list[PORT_SIGMA_POS])
                else:
                    p_out.arr[:] = p_in.arr.copy()

                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("GAUSSIAN BLUR JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_median_blur_func(port_list: list = None) -> bool:
    """
    Main function for blur job.

    :param port_list: Param needed list of port names [input1, wave_offset, kernel_size, sigma, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_POS = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3

    # check if param OK
    if len(port_list) != 4:
        log_error_to_console("MEDIAN BLUR JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                p_out.arr[:] = cv2.medianBlur(src=p_in.arr, ksize=port_list[PORT_KERNEL_POS])
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("MEDIAN BLUR JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_mean_blur_func(port_list: list = None) -> bool:
    """
    Main function for mean filter job.

    :param port_list: Param needed list of port names [input1, wave_offset, kernel_size, sigma, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_POS = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3

    # check if param OK
    if len(port_list) != 4:
        log_error_to_console("MEAN BLUR JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                if port_list[PORT_KERNEL_POS] != 0:
                    p_out.arr[:] = cv2.blur(src=p_in.arr.copy(), ksize=(port_list[PORT_KERNEL_POS], port_list[PORT_KERNEL_POS]))
                else:
                    p_out.arr[:] = p_in.arr.copy()
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("MEAN BLUR JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_unsharp_filter_func(port_list: list = None) -> bool:
    """
    The Unsharp filter can be used to enhance the edges of an image.

    :param port_list: Param needed list of port names [input1,  wave_offset, kernel_size, sigma, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_RADIUS_POS = 2
    # noinspection PyPep8Naming
    PORT_PERCENT_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_POS = 4

    # check if param OK
    if len(port_list) != 5:
        log_error_to_console("UNSHARP FILTER JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                img = Image.fromarray(obj=p_in.arr)
                p_out.arr[:] = np.array(img.filter(filter=ImageFilter.UnsharpMask(radius=port_list[PORT_RADIUS_POS],
                                                                                  percent=port_list[PORT_PERCENT_POS])))
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("UNSHARP FILTER JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_bilateral_blur_func(port_list: list = None) -> bool:
    """
    Main function for blur job.

    :param port_list: Param needed list of port names [port_in_image, wave_offset, distance, sigma_color, sigma_space, port_out_image]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_DISTANCE_VAL = 2
    # noinspection PyPep8Naming
    PORT_SIGMA_COLOR = 3
    # noinspection PyPep8Naming
    PORT_SIGMA_SPACE = 4
    # noinspection PyPep8Naming
    PORT_OUT_POS = 5

    # check if param OK
    if len(port_list) != 6:
        log_error_to_console("BILATERAL FILTER JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                p_out.arr[:] = cv2.bilateralFilter(src=p_in.arr, d=port_list[PORT_DISTANCE_VAL], sigmaColor=port_list[PORT_SIGMA_COLOR],
                                                   sigmaSpace=port_list[PORT_SIGMA_SPACE])
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("BILATERAL FILTER JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


@jit(nopython=True)
def conservative_filter(mat_in: Port.arr, kernel_size: int) -> Port.arr:
    """
    Conservative filter algorithm.
    :param mat_in: original image
    :param kernel_size: kernel size
    :return: new image
    """
    mat_out = mat_in.copy()
    number_row, number_col = mat_in.shape
    temp = []
    indexer = kernel_size // 2

    for i in range(number_row):
        for j in range(number_col):
            for k in range(i - indexer, i + indexer + 1):
                for m in range(j - indexer, j + indexer + 1):
                    if (k > -1) and (k < number_row):
                        if (m > -1) and (m < number_col):
                            temp.append(mat_in[k, m])
            temp.remove(mat_in[i, j])
            max_value = max(temp)
            min_value = min(temp)

            if mat_in[i, j] > max_value:
                mat_out[i, j] = max_value

            elif mat_in[i, j] < min_value:
                mat_out[i, j] = min_value

            temp.clear()

    return mat_out


def main_conservative_filter_func(port_list: list = None) -> bool:
    """
    The conservative filter is used to remove salt and pepper noise. Determines the minimum intensity and maximum intensity
    within a neighborhood of a pixel. If the intensity of the center pixel is greater than the maximum value it is replaced
    by the maximum value. If it is less than the minimum value than it is replaced by the minimum value. The conservative
    filter preserves edges but does not remove speckle noise.

    :param port_list: Param needed list of port names [input1, wave_offset, kernel_size, sigma, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # TODO think maybe use https://pypi.org/project/PyStretch/
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_POS = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3

    # check if param OK
    if len(port_list) != 4:
        log_error_to_console("CONSERVATIVE FILTER JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                p_out.arr[:] = conservative_filter(mat_in=p_in.arr.copy(), kernel_size=port_list[PORT_KERNEL_POS])

                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("CONSERVATIVE FILTER JOB NOK: ", str(error))
                pass

        else:
            return False

        return True


@jit(nopython=True)
def crimmins_func(data: Port.arr) -> Port.arr:
    """
    Crimmins smoothing algorithm
    :param data: original image
    :return: new image
    """
    # Dark pixel adjustment

    # First Step
    # N-S
    new_image = data.copy()
    number_row, number_col = data.shape

    for i in range(1, number_row):
        for j in range(number_col):
            if data[i - 1, j] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(number_row):
        for j in range(number_col - 1):
            if data[i, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, number_row):
        for j in range(1, number_col):
            if data[i - 1, j - 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, number_row):
        for j in range(number_col - 1):
            if data[i - 1, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, number_row - 1):
        for j in range(number_col):
            if (data[i - 1, j] > data[i, j]) and (data[i, j] <= data[i + 1, j]):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(number_row):
        for j in range(1, number_col - 1):
            if (data[i, j + 1] > data[i, j]) and (data[i, j] <= data[i, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, number_row - 1):
        for j in range(1, number_col - 1):
            if (data[i - 1, j - 1] > data[i, j]) and (data[i, j] <= data[i + 1, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, number_row - 1):
        for j in range(1, number_col - 1):
            if (data[i - 1, j + 1] > data[i, j]) and (data[i, j] <= data[i + 1, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, number_row - 1):
        for j in range(number_col):
            if (data[i + 1, j] > data[i, j]) and (data[i, j] <= data[i - 1, j]):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(number_row):
        for j in range(1, number_col - 1):
            if (data[i, j - 1] > data[i, j]) and (data[i, j] <= data[i, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, number_row - 1):
        for j in range(1, number_col - 1):
            if (data[i + 1, j + 1] > data[i, j]) and (data[i, j] <= data[i - 1, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, number_row - 1):
        for j in range(1, number_col - 1):
            if (data[i + 1, j - 1] > data[i, j]) and (data[i, j] <= data[i - 1, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(number_row - 1):
        for j in range(number_col):
            if data[i + 1, j] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(number_row):
        for j in range(1, number_col):
            if data[i, j - 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(number_row - 1):
        for j in range(number_col - 1):
            if data[i + 1, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(number_row - 1):
        for j in range(1, number_col):
            if data[i + 1, j - 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image

    # Light pixel adjustment

    # First Step
    # N-S
    for i in range(1, number_row):
        for j in range(number_col):
            if data[i - 1, j] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(number_row):
        for j in range(number_col - 1):
            if data[i, j + 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, number_row):
        for j in range(1, number_col):
            if data[i - 1, j - 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, number_row):
        for j in range(number_col - 1):
            if data[i - 1, j + 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, number_row - 1):
        for j in range(number_col):
            if (data[i - 1, j] < data[i, j]) and (data[i, j] >= data[i + 1, j]):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(number_row):
        for j in range(1, number_col - 1):
            if (data[i, j + 1] < data[i, j]) and (data[i, j] >= data[i, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, number_row - 1):
        for j in range(1, number_col - 1):
            if (data[i - 1, j - 1] < data[i, j]) and (data[i, j] >= data[i + 1, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, number_row - 1):
        for j in range(1, number_col - 1):
            if (data[i - 1, j + 1] < data[i, j]) and (data[i, j] >= data[i + 1, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, number_row - 1):
        for j in range(number_col):
            if (data[i + 1, j] < data[i, j]) and (data[i, j] >= data[i - 1, j]):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(number_row):
        for j in range(1, number_col - 1):
            if (data[i, j - 1] < data[i, j]) and (data[i, j] >= data[i, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, number_row - 1):
        for j in range(1, number_col - 1):
            if (data[i + 1, j + 1] < data[i, j]) and (data[i, j] >= data[i - 1, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, number_row - 1):
        for j in range(1, number_col - 1):
            if (data[i + 1, j - 1] < data[i, j]) and (data[i, j] >= data[i - 1, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(number_row - 1):
        for j in range(number_col):
            if data[i + 1, j] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(number_row):
        for j in range(1, number_col):
            if data[i, j - 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(number_row - 1):
        for j in range(number_col - 1):
            if data[i + 1, j + 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(number_row - 1):
        for j in range(1, number_col):
            if data[i + 1, j - 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1

    return new_image


def main_crimmins_func(port_list: list = None) -> bool:
    """
    The Crimmins complementary culling algorithm is used to remove speckle noise and smooth the edges.
    It also reduces the intensity of salt and pepper noise. The algorithm compares the intensity of a pixel
    in a image with the intensities of its 8 neighbors. The algorithm considers 4 sets of neighbors
    (N-S, E-W, NW-SE, NE-SW.) Let a,b,c be three consecutive pixels (for example from E-S). .
    :param port_list: Param needed list of port names [input1, wave_offset, kernel_size, sigma, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_OUT_POS = 2

    if len(port_list) != 3:
        log_error_to_console("CRIMMINS FILTER JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                p_out.arr[:] = crimmins_func(data=p_in.arr.copy())
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("CRIMMINS FILTER JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_guided_filter_func(port_list: list = None) -> bool:
    """
    Derived from a local linear model, the guided filter computes the filtering output by considering the content of
    a guidance image, which can be the input image itself or another different image.

    :param port_list: Param needed list of port names [input1, wave_offset, radius_size, eps, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_RAD_POS = 2
    # noinspection PyPep8Naming
    PORT_EPS_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_POS = 4

    # check if param OK
    if len(port_list) != 5:
        log_error_to_console("GUIDED FILTER JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                p_out.arr[:] = cv2.ximgproc.guidedFilter(guide=p_in.arr.copy(), src=p_in.arr.copy(),
                                                         radius=port_list[PORT_RAD_POS],
                                                         eps=port_list[PORT_EPS_POS] * port_list[PORT_EPS_POS] * 1000000, dDepth=-1)

                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("GUIDED FILTER JOB NOK: ", str(error))
                pass

        else:
            return False

        return True


def main_l0_smoothing_filter_func(port_list: list = None) -> bool:
    """
    Global image smoothing via L0 gradient minimization.
    #https://sites.fas.harvard.edu/~cs278/papers/l0.pdf
    :param port_list: Param needed list of port names [input, wave_offset, lambda, kappa, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_LAMBDA_POS = 2
    # noinspection PyPep8Naming
    PORT_KAPPA_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_POS = 4

    # check if param OK
    if len(port_list) != 5:
        log_error_to_console("L0 GRADIENT MINIMIZATION FILTER JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                # noinspection PyArgumentList
                p_out.arr[:] = cv2.ximgproc.l0Smooth(p_in.arr.copy(), p_out.arr, port_list[PORT_LAMBDA_POS], port_list[PORT_KAPPA_POS])
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("L0 GRADIENT MINIMIZATION FILTER JOB NOK: ", str(error))
                pass

        else:
            return False

        return True


def main_anisotropic_diffusion_filter_func(port_list: list = None) -> bool:
    """
    https://www2.eecs.berkeley.edu/Pubs/TechRpts/1988/CSD-88-483.pdf
    https://ieeexplore.ieee.org/document/56205
    :param port_list: Param needed list of port names [input, wave_offset, alpha, kappa, niter, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_ALPHA_POS = 2
    # noinspection PyPep8Naming
    PORT_K_POS = 3
    # noinspection PyPep8Naming
    PORT_NITER_POS = 4
    # noinspection PyPep8Naming
    PORT_OUT_POS = 5

    # check if param OK
    if len(port_list) != 6:
        log_error_to_console("ANISOTROPIC DIFFUSION FILTER JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                if len(p_in.arr.shape) == 2:
                    tmp = cv2.cvtColor(src=p_in.arr, code=cv2.COLOR_GRAY2RGB)
                else:
                    tmp = p_in.arr

                result = cv2.ximgproc.anisotropicDiffusion(src=tmp, alpha=port_list[PORT_ALPHA_POS],
                                                           K=port_list[PORT_K_POS], niters=port_list[PORT_NITER_POS])

                if len(p_in.arr.shape) == 2:
                    result = cv2.cvtColor(src=result, code=cv2.COLOR_BGR2GRAY)

                p_out.arr[:] = result
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("ANISOTROPIC DIFFUSION FILTER JOB NOK: ", str(error))
                pass

        else:
            return False

        return True


def main_sharpen_filter_func(port_list: list = None) -> bool:
    """

    :param port_list: Param needed list of port names [input, wave_offset, kernel, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_SIZE = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3

    # check if param OK
    if len(port_list) != 4:
        log_error_to_console("SHARPEN FILTER JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                if port_list[PORT_KERNEL_SIZE] != 0:
                    kernel_sharpen_1 = (np.ones((port_list[PORT_KERNEL_SIZE], port_list[PORT_KERNEL_SIZE])) * (-1)).astype(np.int8)
                    mid = int((port_list[PORT_KERNEL_SIZE] - 1) / 2)
                    kernel_sharpen_1[mid, mid] = kernel_sharpen_1.size

                    result = cv2.filter2D(p_in.arr, -1, kernel_sharpen_1)
                    p_out.arr[:] = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                else:
                    p_out.arr[:] = p_in.arr.copy()

                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("SHARPEN FILTER JOB NOK: ", str(error))
                pass

        else:
            return False

        return True


def main_motion_blur_filter_func(port_list: list = None) -> bool:
    """

    :param port_list: Param needed list of port names [input, wave_offset, kernel, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_SIZE = 2
    # noinspection PyPep8Naming
    PORT_ANGLE = 3
    # noinspection PyPep8Naming
    PORT_OUT_POS = 4

    # check if param OK
    if len(port_list) != 5:
        log_error_to_console("MOTION BLUR FILTER JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                if port_list[PORT_KERNEL_SIZE] != 0 or port_list[PORT_ANGLE] != 0 :
                    # This generates a matrix of motion blur kernels at any angle. The greater the degree, the higher the blur.
                    M = cv2.getRotationMatrix2D(center=(port_list[PORT_KERNEL_SIZE] / 2, port_list[PORT_KERNEL_SIZE] / 2),
                                                angle=port_list[PORT_ANGLE], scale=1)
                    motion_blur_kernel = np.diag(np.ones(port_list[PORT_KERNEL_SIZE]))
                    motion_blur_kernel = cv2.warpAffine(src=motion_blur_kernel, M=M,
                                                        dsize=(port_list[PORT_KERNEL_SIZE], port_list[PORT_KERNEL_SIZE]))

                    motion_blur_kernel = motion_blur_kernel / port_list[PORT_KERNEL_SIZE]
                    blurred = cv2.filter2D(src=p_in.arr.copy(), ddepth=-1, kernel=motion_blur_kernel)

                    p_out.arr[:] = cv2.normalize(src=blurred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                else:
                    p_out.arr[:] = p_in.arr.copy()
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("MOTION BLUR FILTER JOB NOK: ", str(error))
                pass

        else:
            return False

        return True


if __name__ == "__main__":
    pass
