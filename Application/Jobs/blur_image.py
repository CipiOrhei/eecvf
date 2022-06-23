# noinspection PyPackageRequirements
import cv2
import numpy as np
import scipy
import skimage

from numba import jit

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console
from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file


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
def conservative_filter(mat_in, kernel_size: int):
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
def crimmins_func(data):
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


# define a main function, function that will be executed at the begging of the wave
def main_weiner_func(param_list: list = None) -> bool:
    """
    Main function for {job} calculation job.
    :param param_list: Param needed to respect the following list:
                       [enumerate list]
    :return: True if the job executed OK.
    """
    # variables for position of param needed
    # ex:

    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_SIZE = 2
    # noinspection PyPep8Naming
    PORT_K_VALUE = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3
    # verify that the number of parameters are OK.
    if len(param_list) != 4:
        log_error_to_console("WEINER FILTER JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # https://github.com/tranleanh/wiener-median-comparison/blob/c3f428b3540ec8fea32a87ba88d19f3741f51d79/Wiener_Filter.py#L20
        # https://www.researchgate.net/publication/332574579_Image_Processing_Course_Project_Image_Filtering_with_Wiener_Filter_and_Median_Filter
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_POS])

        if port_in.is_valid() is True:
            if True:
            # try:

                # calculate kernel
                kernel = scipy.signal.gaussian(param_list[PORT_KERNEL_SIZE], param_list[PORT_KERNEL_SIZE] / 3).reshape(param_list[PORT_KERNEL_SIZE], 1)
                kernel = np.dot(kernel, kernel.transpose())
                kernel /= np.sum(kernel)

                # kernel /= np.sum(kernel)
                dummy = np.copy(port_in.arr.copy())
                dummy = np.fft.fft2(dummy)
                kernel = np.fft.fft2(kernel, s=port_in.arr.shape)
                kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + param_list[PORT_K_VALUE])
                dummy = dummy * kernel
                dummy = np.abs(np.fft.ifft2(dummy))


                p_out.arr[:] = dummy
                p_out.set_valid()

            # except BaseException as error:
            #     log_error_to_console("WEINER FILTER JOB NOK: ", str(error))
            #     pass
        else:
            return False

        return True


############################################################################################################################################
# Job create functions
############################################################################################################################################

def do_weiner_filter_job(port_input_name: str,
                          kernel_size: int = 5, K_value: int = 10,
                          port_output_name: str = None,
                          wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    TBD
    https://arxiv.org/pdf/1004.5538
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel_size: size of kernel
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'WEINER_FILER_S_{}_K_{}_{}'.format(kernel_size, K_value, port_input_name)

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel_size, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Weiner filter', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               S=str(kernel_size), K=str(K_value))

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_weiner_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


if __name__ == "__main__":
    pass
