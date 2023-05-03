# import what you need
import os

from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console, log_setup_info_to_console
import config_main as CONFIG
from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file
from Utils.plotting import plot_histogram_grey_image
from Application.Config.job_create import custom_kernels_used
import cv2
import numpy as np
import Application.Jobs.kernels
from PIL import ImageFilter, Image
from Application.Jobs.kernels import dilate
import pywt.data
import math

"""
Module handles sharpening algorithm for an image jobs for the APPL block.
"""

############################################################################################################################################
# Internal functions
############################################################################################################################################

def um(img, kernel, strength):
    lap = cv2.filter2D(src=img.copy(), ddepth=cv2.CV_64F, kernel=kernel).astype('int32')
    a_lap = strength * lap
    img = np.float64(img.copy())
    sharp = img - a_lap
    sharp[sharp > 255] = 255
    sharp[sharp < 0] = 0
    return sharp

############################################################################################################################################
# Init functions
############################################################################################################################################

# define a init function, function that will be executed at the begging of the wave
def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :return: INIT or NOT_INIT state for the job
    """
    # log_to_file('DATA YOU NEED TO SAVE EVERY FRAME IN CSV')
    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################

# define a main function, function that will be executed at the begging of the wave
def main_func_histogram_equalization(param_list: list = None) -> bool:
    """
    Main function for histogram equalization calculation job.
    :param param_list: Param needed to respect the following list:
                       [port in, port in wave, save histogram, port out]
    :return: True if the job executed OK.
    """
    # variables for position of param needed
    # ex:

    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_OUT_POS = 2
    # noinspection PyPep8Naming
    PORT_SAVE_HIST = 3
    # verify that the number of parameters are OK.
    if len(param_list) != 4:
        log_error_to_console("HISTOGRAM EQUALIZATION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_POS])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                if len(port_in.arr.shape) == 2:
                    p_out.arr[:] = cv2.equalizeHist(port_in.arr.copy())
                else:
                    ycrcb = cv2.cvtColor(port_in.arr.copy(), cv2.COLOR_BGR2YCR_CB)
                    channels = cv2.split(ycrcb)
                    cv2.equalizeHist(channels[0], channels[0])
                    cv2.merge(channels, ycrcb)
                    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, p_out.arr)
                    # p_out.arr[:]
                if param_list[PORT_SAVE_HIST] == True:
                    plot_histogram_grey_image(image=port_in.arr.copy(), name_folder=port_in.name, picture_name='HIST_EQUAL_' + global_var_handler.PICT_NAME.split('.')[0],
                                              to_save=True, to_show=False)
                    plot_histogram_grey_image(image=p_out.arr.copy(), name_folder=p_out.name, picture_name='HIST_EQUAL_' + global_var_handler.PICT_NAME.split('.')[0],
                                              to_save=True, to_show=False)
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("HISTOGRAM EQUALIZATION JOB NOK: ", str(error))
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
    PORT_KERNEL = 2
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
                if port_list[PORT_KERNEL] == None:
                    kernel = None
                elif 'xy' in port_list[PORT_KERNEL]:
                    kernel = eval('Application.Jobs.kernels.' + port_list[PORT_KERNEL])
                else:
                    kernel = np.array(eval(port_list[PORT_KERNEL]))

                if kernel is not None:
                    kernel_identity = (np.zeros(kernel.shape)).astype(np.int8)
                    mid = int((kernel_identity.shape[0] - 1) / 2)
                    kernel_identity[mid, mid] = 1

                    kernel_high_pass = kernel_identity - kernel
                    img = p_in.arr.copy().astype(np.int16)
                    result = cv2.filter2D(img, cv2.CV_16S, kernel_high_pass)
                    result[result > 255] = 255
                    result[result < 0] = 0
                    # x = np.sum(kernel_high_pass)
                    # tmp = np.sum(kernel_high_pass, where=kernel_high_pass>0)
                    # result = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                    p_out.arr[:] = result
                else:
                    p_out.arr[:] = p_in.arr[:]
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("SHARPEN FILTER JOB NOK: ", str(error))
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
    PORT_THR_POS = 4
    # noinspection PyPep8Naming
    PORT_OUT_POS = 5

    # check if param OK
    if len(port_list) != 6:
        log_error_to_console("UNSHARP FILTER JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                img = Image.fromarray(obj=p_in.arr.copy())
                p_out.arr[:] = np.array(img.filter(filter=ImageFilter.UnsharpMask(radius=port_list[PORT_RADIUS_POS],
                                                                                  percent=port_list[PORT_PERCENT_POS],
                                                                                  threshold=port_list[PORT_THR_POS])))
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("UNSHARP FILTER JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_unsharp_filter_func_long(port_list: list = None) -> bool:
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
    PORT_KERNEL_POS = 2
    # noinspection PyPep8Naming
    PORT_STREGHT_POS = 3
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
                if port_list[PORT_KERNEL_POS] == None:
                    kernel = None
                elif 'xy' in port_list[PORT_KERNEL_POS]:
                    kernel = eval('Application.Jobs.kernels.' + port_list[PORT_KERNEL_POS])
                else:
                    kernel = np.array(eval(port_list[PORT_KERNEL_POS]))

                if kernel is not None:
                    if port_list[PORT_STREGHT_POS] < 0:
                        port_list[PORT_STREGHT_POS] = 1
                    elif port_list[PORT_STREGHT_POS] > 9:
                        port_list[PORT_STREGHT_POS] = 9

                    lap = cv2.filter2D(src=p_in.arr.copy(), ddepth=cv2.CV_16S, kernel=kernel)
                    a_lap = port_list[PORT_STREGHT_POS] * lap
                    img = np.float64(p_in.arr.copy())
                    sharp = img - a_lap
                    sharp[sharp > 255] = 255
                    sharp[sharp < 0] = 0
                    p_out.arr[:] = sharp
                else:
                    p_out.arr[:] = p_in.arr[:]
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("UNSHARP FILTER JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_n_nun_func_long(port_list: list = None) -> bool:
    """
    :param port_list: Param needed list of port names [input1,  wave_offset, kernel_size, sigma, output]
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
    PORT_STREGHT_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_POS = 4

    # check if param OK
    if len(port_list) != 5:
        log_error_to_console("N-NUM JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                if port_list[PORT_KERNEL_POS] == None:
                    kernel = None
                elif 'xy' in port_list[PORT_KERNEL_POS]:
                    kernel = eval('Application.Jobs.kernels.' + port_list[PORT_KERNEL_POS])
                else:
                    kernel = np.array(eval(port_list[PORT_KERNEL_POS]))

                if kernel is not None:
                    if port_list[PORT_STREGHT_POS] < 0:
                        port_list[PORT_STREGHT_POS] = 1
                    elif port_list[PORT_STREGHT_POS] > 9:
                        port_list[PORT_STREGHT_POS] = 9

                    in_img = p_in.arr.copy()

                    if len(p_in.arr.shape) == 3:
                        img_ycbcr = cv2.cvtColor(p_in.arr, cv2.COLOR_BGR2YCR_CB)
                        in_img = img_ycbcr[:, :, 0]

                    # obtain the output of the quadratic filter
                    qf = cv2.filter2D(src=in_img.copy(), ddepth=cv2.CV_16S, kernel=kernel)
                    # obtain the sign of the filter output
                    sign_qf = np.sign(qf)

                    max_qf = np.max(np.abs(qf))

                    nqf = sign_qf * (qf/max_qf)**2
                    nqf = nqf * in_img

                    a_lap = port_list[PORT_STREGHT_POS] * nqf

                    sharp = in_img - a_lap

                    sharp[sharp > 255] = 255
                    sharp[sharp < 0] = 0

                    if len(p_in.arr.shape) == 3:
                        img_ycbcr[:, :, 0] = sharp

                        sharp = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2BGR)

                    p_out.arr[:] = sharp
                else:
                    p_out.arr[:] = p_in.arr[:]
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("N-NUM JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


from numba import jit

@jit(nopython=True)
def fusion_test(coef_list, octavs, fuison_logic):
    LL = np.zeros_like(coef_list[0][0])
    LH = np.zeros_like(coef_list[0][1][0])
    HL = np.zeros_like(coef_list[0][1][1])
    HH = np.zeros_like(coef_list[0][1][2])

    if fuison_logic == 'max':
        LL, (LH, HL, HH) = fusion_max(coef_list, octavs)

    elif fuison_logic == 'average_1':
        for i in range(coef_list[0][1][0].shape[0]):
            for j in range(coef_list[0][1][0].shape[1]):
                for k in range(octavs):
                    param = k
                    LL[i][j] += coef_list[k][0][i][j] * (param)
                    LH[i][j] += coef_list[k][1][0][i][j] * (param)
                    HL[i][j] += coef_list[k][1][1][i][j] * (param)
                    HH[i][j] += coef_list[k][1][2][i][j] * (param)

                LL[i][j] /= octavs
                LH[i][j] /= octavs
                HL[i][j] /= octavs
                HH[i][j] /= octavs

    elif fuison_logic == 'average_2':
        for i in range(coef_list[0][1][0].shape[0]):
            for j in range(coef_list[0][1][0].shape[1]):
                for k in range(octavs):
                    param = octavs - k
                    LL[i][j] += coef_list[k][0][i][j] * (param)
                    LH[i][j] += coef_list[k][1][0][i][j] * (param)
                    HL[i][j] += coef_list[k][1][1][i][j] * (param)
                    HH[i][j] += coef_list[k][1][2][i][j] * (param)

                LL[i][j] /= octavs
                LH[i][j] /= octavs
                HL[i][j] /= octavs
                HH[i][j] /= octavs

    elif fuison_logic == 'average_3':
        for i in range(coef_list[0][1][0].shape[0]):
            for j in range(coef_list[0][1][0].shape[1]):
                for k in range(octavs):
                    param = 1
                    LL[i][j] += coef_list[k][0][i][j] * (param)
                    LH[i][j] += coef_list[k][1][0][i][j] * (param)
                    HL[i][j] += coef_list[k][1][1][i][j] * (param)
                    HH[i][j] += coef_list[k][1][2][i][j] * (param)

                LL[i][j] /= octavs
                LH[i][j] /= octavs
                HL[i][j] /= octavs
                HH[i][j] /= octavs

    return LL, (LH, HL, HH)


def main_um_dilated_2dwt(port_list: list = None) -> bool:
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
    PORT_KERNEL_POS = 2
    # noinspection PyPep8Naming
    PORT_STREGHT_POS = 3
    # noinspection PyPep8Naming
    PORT_FUSION_LVL = 4
    # noinspection PyPep8Naming
    PORT_WAVELENGT = 5
    # noinspection PyPep8Naming
    PORT_OUT_POS = 6
    # noinspection PyPep8Naming
    PORT_FUSION_POS = 7

    # check if param OK
    if len(port_list) != 8:
        log_error_to_console("UM DILATED 2DWT JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            # try:
            if True:
                if port_list[PORT_KERNEL_POS] == None:
                    kernel = None
                elif 'xy' in port_list[PORT_KERNEL_POS]:
                    kernel = eval('Application.Jobs.kernels.' + port_list[PORT_KERNEL_POS])
                else:
                    kernel = np.array(eval(port_list[PORT_KERNEL_POS]))

                if kernel is not None:
                    if port_list[PORT_STREGHT_POS] < 0:
                        port_list[PORT_STREGHT_POS] = 1
                    elif port_list[PORT_STREGHT_POS] > 9:
                        port_list[PORT_STREGHT_POS] = 9

                    coef_list = list()

                    in_img = p_in.arr.copy()

                    if len(p_in.arr.shape) == 3:
                        img_ycbcr = cv2.cvtColor(p_in.arr, cv2.COLOR_BGR2YCR_CB)
                        in_img = img_ycbcr[:, :, 0]

                    for l in range(port_list[PORT_FUSION_LVL]):
                        res_um = um(img=in_img, kernel=dilate(kernel, l), strength=port_list[PORT_STREGHT_POS])
                        coef_list.append(pywt.dwt2(res_um.astype('int32'), port_list[PORT_WAVELENGT]))


                    coeffs = fusion_test(coef_list=coef_list, octavs=port_list[PORT_FUSION_LVL], fuison_logic=port_list[PORT_FUSION_POS])

                    inverse = pywt.idwt2(coeffs, port_list[PORT_WAVELENGT])

                    inverse = cv2.normalize(src=inverse, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

                    if len(p_in.arr.shape) == 3:
                        img_ycbcr[:, :, 0] = inverse

                        inverse = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2BGR)

                    p_out.arr[:] = inverse
                else:
                    p_out.arr[:] = p_in.arr[:]
                p_out.set_valid()
            # except BaseException as error:
            #     log_error_to_console("UM DILATED 2DWT JOB NOK: ", str(error))
            #     pass
        else:
            return False

        return True


@jit(nopython=True)
def fusion_max(coef_list, octavs):
    LL = np.zeros_like(coef_list[0][0])
    LH = np.zeros_like(coef_list[0][1][0])
    HL = np.zeros_like(coef_list[0][1][1])
    HH = np.zeros_like(coef_list[0][1][2])

    for i in range(coef_list[0][1][0].shape[0]):
        for j in range(coef_list[0][1][0].shape[1]):
            l_ll = list()
            l_lh = list()
            l_hl = list()
            l_hh = list()

            for k in range(octavs):
                l_ll.append(coef_list[k][0][i][j])
                l_lh.append(coef_list[k][1][0][i][j])
                l_hl.append(coef_list[k][1][1][i][j])
                l_hh.append(coef_list[k][1][2][i][j])

            LL[i][j] = max(l_ll)
            LH[i][j] = max(l_lh)
            HL[i][j] = max(l_hl)
            HH[i][j] = max(l_hh)

    return LL, (LH, HL, HH)


def main_um_2dwt_fusion(port_list: list = None) -> bool:
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
    PORT_O_POS = 2
    # noinspection PyPep8Naming
    PORT_M_POS = 3
    # noinspection PyPep8Naming
    PORT_K_LVL = 4
    # noinspection PyPep8Naming
    PORT_S_LVL = 5
    # noinspection PyPep8Naming
    PORT_WAVELET = 6
    # noinspection PyPep8Naming
    PORT_OUT_POS = 7

    # check if param OK
    if len(port_list) != 8:
        log_error_to_console("UM 2DWT FUSION JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            # try:
            if True:
                coef_list = list()

                in_img = p_in.arr.copy()

                if len(p_in.arr.shape) == 3 :
                    img_ycbcr = cv2.cvtColor(p_in.arr, cv2.COLOR_BGR2YCR_CB)
                    in_img = img_ycbcr[:,:,0]

                for octav in range(port_list[PORT_O_POS]):
                    arg = port_list[PORT_S_LVL] * port_list[PORT_K_LVL] ** octav
                    img_blur = cv2.GaussianBlur(src=in_img.copy(), ksize=(0, 0), sigmaX=arg)

                    res_um = (port_list[PORT_M_POS] + 1) * in_img.copy().astype('int32') - port_list[PORT_M_POS] * img_blur.astype('int32')

                    coef_list.append(pywt.dwt2(res_um, port_list[PORT_WAVELET]))


                coeffs = fusion_max(coef_list, port_list[PORT_O_POS])
                inverse = pywt.idwt2(coeffs, port_list[PORT_WAVELET])

                inverse = cv2.normalize(src=inverse, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

                if len(p_in.arr.shape) == 3:
                    img_ycbcr[:, :, 0] = inverse

                    inverse = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2BGR)


                p_out.arr[:] = inverse

                p_out.set_valid()
            # except BaseException as error:
            #     log_error_to_console("UM DILATED 2DWT JOB NOK: ", str(error))
            #     pass
        else:
            return False

        return True


@jit(nopython=True)
def clipping(img, img_filtered, M, N, win_offset):
    for i in np.arange(0, N):
        xleft = i - win_offset
        xright = i + win_offset

        if xleft < 0:
            xleft = 0
        if xright >= N:
            xright = N

        for j in np.arange(0, M):
            yup = j - win_offset
            ydown = j + win_offset

            if yup < 0:
                yup = 0
            if ydown >= M:
                ydown = M

            # assert_indices_in_range(N, M, xleft, xright, yup, ydown)
            window = img[xleft:xright, yup:ydown]
            min = np.min(window)
            max = np.max(window)

            if img_filtered[i, j] > max:
                img_filtered[i, j] = max
            elif img_filtered[i, j] < min:
                img_filtered[i, j] = min
            else:
                img_filtered[i, j] = img_filtered[i, j]

    return img_filtered


def main_constrained_unsharp_filter(port_list: list = None) -> bool:
    """
        :param port_list: Param needed list of port names [input1,  wave_offset, kernel_size, sigma, output]
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
    PORT_STREGHT_POS = 3
    # noinspection PyPep8Naming
    PORT_LS_WIN = 4
    # noinspection PyPep8Naming
    PORT_LS_VAL = 5
    # noinspection PyPep8Naming
    PORT_CLIPPING_TH = 6
    # noinspection PyPep8Naming
    PORT_CASCADE_VER = 7
    # noinspection PyPep8Naming
    PORT_OUT_POS = 8

    # check if param OK
    if len(port_list) != 9:
        log_error_to_console("CONSTRAINED UM JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            if True:
            # try:
                if port_list[PORT_KERNEL_POS] == None:
                    kernel = None
                elif 'xy' in port_list[PORT_KERNEL_POS]:
                    kernel = eval('Application.Jobs.kernels.' + port_list[PORT_KERNEL_POS])
                else:
                    kernel = np.array(eval(port_list[PORT_KERNEL_POS]))

                if kernel is not None:
                    if port_list[PORT_STREGHT_POS] < 0:
                        port_list[PORT_STREGHT_POS] = 0.01
                    elif port_list[PORT_STREGHT_POS] > 1:
                        port_list[PORT_STREGHT_POS] = 1

                    if len(p_in.arr.shape) > 2:
                        img_yuv = cv2.cvtColor(p_in.arr.copy(), cv2.COLOR_BGR2YUV)
                        img = img_yuv[:, :, 0]
                    else:
                        img = p_in.arr.copy()

                    from Application.Jobs.blur_image import lee_sigma_filter_processing

                    # we process the entire img as float64 to avoid type overflow error
                    img_filtered = np.zeros_like(img)

                    img_f = lee_sigma_filter_processing(img_filtered=img_filtered, img=img, win_offset=int(port_list[PORT_LS_WIN] / 2),
                                                        M=img.shape[0], N=img.shape[1], sigma=port_list[PORT_LS_VAL]).astype('int16')

                    if port_list[PORT_CASCADE_VER]:
                        img_h = cv2.filter2D(src=img_f.copy(), ddepth=cv2.CV_16S, kernel=kernel)
                    else:
                        img_h = cv2.filter2D(src=img.copy(), ddepth=cv2.CV_16S, kernel=kernel)

                    img_h = port_list[PORT_STREGHT_POS] * img_h

                    img_s = img_f - img_h
                    img_s[img_s > 255] = 255
                    img_s[img_s < 0] = 0

                    sharp = clipping(img=img_f, img_filtered=img_s, M=img.shape[1], N=img.shape[0], win_offset=port_list[PORT_CLIPPING_TH])

                    if len(p_in.arr.shape) > 2:
                        img_yuv[:, :, 0] = sharp
                        sharp = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                    p_out.arr[:] = sharp
                else:
                    p_out.arr[:] = p_in.arr[:]
                p_out.set_valid()
            # except BaseException as error:
            #     log_error_to_console("UNSHARP FILTER JOB NOK: ", str(error))
            #     pass
        else:
            return False

        return True


def main_nonlinear_unsharp_filter(port_list: list = None) -> bool:
    """
        :param port_list: Param needed list of port names [input1,  wave_offset, kernel_size, sigma, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_GAUSS_KERNEL_SIZE = 2
    # noinspection PyPep8Naming
    PORT_GAUSS_SIGMA = 3
    # noinspection PyPep8Naming
    PORT_FIRST_ORDER_X = 4
    # noinspection PyPep8Naming
    PORT_FIRST_ORDER_Y = 5
    # noinspection PyPep8Naming
    PORT_LAPLACE_KERNEL = 6
    # noinspection PyPep8Naming
    PORT_HPF_STRENGHT = 7
    # noinspection PyPep8Naming
    PORT_OUT_POS = 8

    # check if param OK
    if len(port_list) != 9:
        log_error_to_console("NONLINEAER UM JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            if True:
            # try:

                if 'x' in port_list[PORT_FIRST_ORDER_X] or 'y' in port_list[PORT_FIRST_ORDER_Y]:
                    kernel_x = eval('Application.Jobs.kernels.' + port_list[PORT_FIRST_ORDER_X])
                    kernel_y = eval('Application.Jobs.kernels.' + port_list[PORT_FIRST_ORDER_Y])
                else:
                    kernel_x = np.array(eval(port_list[PORT_FIRST_ORDER_X]))
                    kernel_y = np.array(eval(port_list[PORT_FIRST_ORDER_Y]))

                if port_list[PORT_LAPLACE_KERNEL] == None:
                    kernel = None
                elif 'xy' in port_list[PORT_LAPLACE_KERNEL]:
                    kernel = eval('Application.Jobs.kernels.' + port_list[PORT_LAPLACE_KERNEL])
                else:
                    kernel = np.array(eval(port_list[PORT_LAPLACE_KERNEL]))

                if (kernel is not None) or (kernel_x is not None) or (kernel_y is not None):
                    if port_list[PORT_HPF_STRENGHT] < 0:
                        port_list[PORT_HPF_STRENGHT] = 0.01
                    elif port_list[PORT_HPF_STRENGHT] > 1:
                        port_list[PORT_HPF_STRENGHT] = 1

                    if len(p_in.arr.shape) > 2:
                        img_yuv = cv2.cvtColor(p_in.arr.copy(), cv2.COLOR_BGR2YUV)
                        img = img_yuv[:, :, 0]
                    else:
                        img = p_in.arr.copy()

                    # Apply a Gaussian filter
                    img_blur = cv2.GaussianBlur(src=img.copy(), ksize=(port_list[PORT_GAUSS_KERNEL_SIZE], port_list[PORT_GAUSS_KERNEL_SIZE]),
                                                sigmaX=port_list[PORT_GAUSS_SIGMA], sigmaY=port_list[PORT_GAUSS_SIGMA])

                    # Magnitude matrices for Ix/dx and Iy/dy
                    magnitude_x = np.zeros(shape=img.shape, dtype=np.float32)
                    magnitude_y = np.zeros(shape=img.shape, dtype=np.float32)

                    # flip kernels for a real convolution to be done by cv2.filter2D
                    kernel_x = kernel_x[::-1, ::-1]
                    kernel_y = kernel_y[::-1, ::-1]

                    cv2.filter2D(src=img.copy(), ddepth=cv2.CV_32F, kernel=kernel_x, dst=magnitude_x, anchor=(-1, -1)).astype('int32')
                    cv2.filter2D(src=img.copy(), ddepth=cv2.CV_32F, kernel=kernel_y, dst=magnitude_y, anchor=(-1, -1)).astype('int32')

                    result_first_order = np.hypot(magnitude_x, magnitude_y).astype('int32')

                    img_h = cv2.filter2D(src=img.copy(), ddepth=cv2.CV_16S, kernel=kernel).astype('int32')

                    sub_operator = cv2.multiply(result_first_order, img_h)

                    sharp = img_blur - port_list[PORT_HPF_STRENGHT] * sub_operator

                    sharp[sharp > 255] = 255
                    sharp[sharp < 0] = 0

                    if len(p_in.arr.shape) > 2:
                        img_yuv[:, :, 0] = sharp
                        sharp = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                    p_out.arr[:] = sharp
                else:
                    p_out.arr[:] = p_in.arr[:]
                p_out.set_valid()
            # except BaseException as error:
            #     log_error_to_console("UNSHARP FILTER JOB NOK: ", str(error))
            #     pass
        else:
            return False

        return True


def main_adaptive_num_func_long(port_list: list = None) -> bool:
    """
    :param port_list: Param needed list of port names [input1,  wave_offset, kernel_size, sigma, output]
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
    PORT_DISTANCE_VAL = 3
    # noinspection PyPep8Naming
    PORT_SIGMA_COLOR = 4
    # noinspection PyPep8Naming
    PORT_SIGMA_SPACE = 5
    # noinspection PyPep8Naming
    PORT_T1_POS = 6
    # noinspection PyPep8Naming
    PORT_T2_POS = 7
    # noinspection PyPep8Naming
    PORT_OUT_POS = 8

    # check if param OK
    if len(port_list) != 9:
        log_error_to_console("ANUM JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                if port_list[PORT_KERNEL_POS] == None:
                    kernel = None
                elif 'xy' in port_list[PORT_KERNEL_POS]:
                    kernel = eval('Application.Jobs.kernels.' + port_list[PORT_KERNEL_POS])
                else:
                    kernel = np.array(eval(port_list[PORT_KERNEL_POS]))

                if kernel is not None:

                    in_img = p_in.arr.copy()

                    if len(p_in.arr.shape) == 3:
                        img_ycbcr = cv2.cvtColor(p_in.arr, cv2.COLOR_BGR2YCR_CB)
                        in_img = img_ycbcr[:, :, 0]

                    lp = cv2.bilateralFilter(src=in_img.copy(), d=port_list[PORT_DISTANCE_VAL], sigmaColor=port_list[PORT_SIGMA_COLOR], sigmaSpace=port_list[PORT_SIGMA_SPACE])

                    # obtain the output of the quadratic filter
                    hf = cv2.filter2D(src=in_img.copy(), ddepth=cv2.CV_16S, kernel=kernel)
                    # obtain the sign of the filter output

                    def piecewise_theta(gh, T1, T2):
                        # if gh <= T1:
                        #     return gh / (T2-T1)
                        # elif T1 < gh <= T2:
                        #     return 1
                        # else:
                        #     return  (T2-T1) / (gh)

                        if gh <= T1:
                            return (gh / T1)
                        elif T1 < gh <= T2:
                            return 1
                        else:
                            return (255 - gh) / (255 - T2)

                    alpha = np.vectorize(piecewise_theta)(np.abs(hf), T1=port_list[PORT_T1_POS], T2=port_list[PORT_T2_POS])

                    a_lap = alpha * hf

                    sharp = lp - a_lap

                    sharp[sharp > 255] = 255
                    sharp[sharp < 0] = 0

                    if len(p_in.arr.shape) == 3:
                        img_ycbcr[:, :, 0] = sharp

                        sharp = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2BGR)

                    p_out.arr[:] = sharp
                else:
                    p_out.arr[:] = p_in.arr[:]
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("ANUM JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


@jit(nopython=True)
def local_mean(img, kernel_size=3):
    img = img.astype(np.float32)
    height, width = img.shape
    mean = np.zeros_like(img)

    for i in range(height):
        for j in range(width):
            # Get coordinates of the 3x3 block
            i_min = max(0, i - kernel_size // 2)
            i_max = min(height, i + kernel_size // 2 + 1)
            j_min = max(0, j - kernel_size // 2)
            j_max = min(width, j + kernel_size // 2 + 1)

            # Calculate mean for the block
            block = img[i_min:i_max, j_min:j_max]
            mean[i, j] = np.mean(block)

    return mean

@jit(nopython=True)
def local_variance(img, kernel_size=3):
    img = img.astype(np.float32)
    height, width = img.shape
    variance = np.zeros_like(img)
    mean = local_mean(img, kernel_size=kernel_size)

    for i in range(height):
        for j in range(width):
            # Get coordinates of the 3x3 block
            i_min = max(0, i - kernel_size // 2)
            i_max = min(height, i + kernel_size // 2 + 1)
            j_min = max(0, j - kernel_size // 2)
            j_max = min(width, j + kernel_size // 2 + 1)

            # Calculate variance for the block
            block = img[i_min:i_max, j_min:j_max]
            block_mean = mean[i, j]
            variance[i, j] = np.mean((block - block_mean) ** 2)

    return variance


def variable_gain(v, t1, t2, a_2, a_3, a_1=1):
    if v < t1:
        return a_1
    elif v < t2:
        return a_2
    else:
        return a_3


@jit(nopython=True)
def calculate_R(in_img, beta, z):
    R = np.zeros_like(in_img, dtype=np.float64)

    for i in range(in_img.shape[0]):
        for j in range(1, in_img.shape[1] - 1, 1):
            R[i, j] = (1 - beta) * R[i, j - 1] + beta * z[i, j] * z[i,j]

    return R


@jit(nopython=True)
def calculate_lambda(in_img, gd, gx, z, R, rho):
    lambda_gain = np.zeros_like(in_img, dtype=np.float64)
    # Step 5: Adaptive Gauss-Newton algorithm
    for i in range(in_img.shape[0]):
        for j in range(1, in_img.shape[1] - 1, 1):
            # Update lambda
            e = gd[i, j] - (gx[i, j] - lambda_gain[i, j] * z[i, j])

            # t = lambda_gain[i, j] + 2 * rho * e * z[i,j] * invert_R[i, j]

            if abs(R[i, j]) < 1e-8:
                t = lambda_gain[i, j]
            else:
                t = lambda_gain[i, j] + 2 * rho * e * z[i, j] * (1 / R[i, j])

            lambda_gain[i, j + 1] = t

    return lambda_gain

def main_adaptive_um(port_list: list = None) -> bool:
    """
    :param port_list: Param needed list of port names [input1,  wave_offset, kernel_size, sigma, output]
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
    PORT_T1 = 3
    # noinspection PyPep8Naming
    PORT_T2 = 4
    # noinspection PyPep8Naming
    PORT_ALPHA_LOW = 5
    # noinspection PyPep8Naming
    PORT_ALPHA_HIGH = 6
    # noinspection PyPep8Naming
    PORT_RHO = 7
    # noinspection PyPep8Naming
    PORT_BETA = 8
    # noinspection PyPep8Naming
    PORT_OUT_POS = 9

    # check if param OK
    if len(port_list) != 10:
        log_error_to_console("AUM JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                if port_list[PORT_KERNEL_POS] == None:
                    kernel = None
                elif 'xy' in port_list[PORT_KERNEL_POS]:
                    kernel = eval('Application.Jobs.kernels.' + port_list[PORT_KERNEL_POS])
                else:
                    kernel = np.array(eval(port_list[PORT_KERNEL_POS]))

                if kernel is not None:

                    in_img = p_in.arr.copy()

                    if len(p_in.arr.shape) == 3:
                        img_ycbcr = cv2.cvtColor(p_in.arr, cv2.COLOR_BGR2YCR_CB)
                        in_img = img_ycbcr[:, :, 0]

                    # Step 1: Compute the Z, Laplacian of the image
                    z = cv2.filter2D(src=in_img.copy(), ddepth=cv2.CV_16S, kernel=kernel)

                    # Step 2: Compute local variance
                    v = local_variance(img=in_img.copy())
                    alpha = np.vectorize(variable_gain)(v, t1=port_list[PORT_T1], t2=port_list[PORT_T2], a_2=port_list[PORT_ALPHA_LOW], a_3=port_list[PORT_ALPHA_HIGH])

                    # Step 3: Compute the desired local dynamics
                    g_operator = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                    gx = cv2.filter2D(src=in_img.copy(), ddepth=cv2.CV_16S, kernel=g_operator)
                    gd = gx * alpha

                    # Step 4: Initialize R and calculate
                    R = calculate_R(in_img=in_img, beta=port_list[PORT_BETA], z=z)

                    # Step 5: Adaptive Gauss-Newton algorithm
                    lambda_gain = calculate_lambda(in_img=in_img, gd=gd, gx=gx, z=z, R=R, rho=port_list[PORT_RHO])

                    lambda_gain = cv2.normalize(src=lambda_gain, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    enhancement = lambda_gain * z

                    img_out = in_img - enhancement

                    img_out = np.clip(img_out, 0, 255)

                    if len(p_in.arr.shape) == 3:
                        img_ycbcr[:, :, 0] = img_out
                        img_out = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2BGR)

                    p_out.arr[:] = img_out
                else:
                    p_out.arr[:] = p_in.arr[:]
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("AUM JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


@jit(nopython=True)
def calculate_rho(V, S, kernel_size=3):
    V = V.astype(np.float32)
    height, width = V.shape
    rho = np.zeros_like(V)

    for i in range(height):
        for j in range(width):
            # Get coordinates of the 3x3 block
            i_min = max(0, i - kernel_size // 2)
            i_max = min(height, i + kernel_size // 2 + 1)
            j_min = max(0, j - kernel_size // 2)
            j_max = min(width, j + kernel_size // 2 + 1)

            # Calculate mean for the block
            x = np.sum(V[i_min:i_max, j_min:j_max] * S[i_min:i_max, j_min:j_max])
            y = math.sqrt(np.sum(np.power(V[i_min:i_max, j_min:j_max],2))) * math.sqrt(np.sum(np.power(S[i_min:i_max, j_min:j_max],2)))

            if y == 0:
                rho[i,j] = x
            else:
                rho[i, j] = x/y

    return rho


@jit(nopython=True)
def local_variance(img, kernel_size=3):
    img = img.astype(np.float32)
    height, width = img.shape
    mean = np.zeros_like(img)
    variance = np.zeros_like(img)

    for i in range(height):
        for j in range(width):
            # Get coordinates of the kernel_size x kernel_size block
            i_min = max(0, i - kernel_size // 2)
            i_max = min(height, i + kernel_size // 2 + 1)
            j_min = max(0, j - kernel_size // 2)
            j_max = min(width, j + kernel_size // 2 + 1)

            # Calculate mean for the block
            block = img[i_min:i_max, j_min:j_max]
            mean[i, j] = np.mean(block)
            # Calculate variance for the block
            variance[i, j] = np.var(block)

    return variance


def main_selective_um(port_list: list = None) -> bool:
    """
    :param port_list: Param needed list of port names [input1,  wave_offset, kernel_size, sigma, output]
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
    PORT_DISTANCE_VAL = 3
    # noinspection PyPep8Naming
    PORT_SIGMA_COLOR = 4
    # noinspection PyPep8Naming
    PORT_SIGMA_SPACE = 5
    # noinspection PyPep8Naming
    PORT_L_V_POS = 6
    # noinspection PyPep8Naming
    PORT_L_S_POS = 7
    # noinspection PyPep8Naming
    PORT_T_POS = 8
    # noinspection PyPep8Naming
    PORT_OUT_POS = 9

    # check if param OK
    if len(port_list) != 10:
        log_error_to_console("SUM JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                if port_list[PORT_KERNEL_POS] == None:
                    kernel = None
                elif 'xy' in port_list[PORT_KERNEL_POS]:
                    kernel = eval('Application.Jobs.kernels.' + port_list[PORT_KERNEL_POS])
                else:
                    kernel = np.array(eval(port_list[PORT_KERNEL_POS]))

                if kernel is not None:
                    # Step 0: convert image
                    if len(p_in.arr.shape) == 3:
                        # Convert the BRG image to RGB
                        hsv_img = cv2.cvtColor(p_in.arr.copy(), cv2.COLOR_BGR2HSV)
                    else:
                        hsv_img = cv2.cvtColor(cv2.cvtColor(p_in.arr.copy(), cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

                    # Step 1: convert the image in the HSV domain
                    H_in, S_in, V_in = cv2.split(hsv_img)

                    V_bf = cv2.bilateralFilter(src=V_in.copy(), d=port_list[PORT_DISTANCE_VAL], sigmaColor=port_list[PORT_SIGMA_COLOR], sigmaSpace=port_list[PORT_SIGMA_SPACE])

                    # Step 3: Calculate the average of V
                    V_in_mean = local_mean(img=V_in.copy())
                    V_in_var = V_in - V_in_mean

                    # Step 4: Calculate the average of S
                    S_in_mean = local_mean(img=S_in.copy())
                    S_in_var = S_in - S_in_mean

                    # Step 5: calculate rho
                    rho = calculate_rho(V=V_in_var, S=S_in_var)

                    # Step 6: Calculate UM and UMS
                    V_in_hpf = cv2.filter2D(src=V_in.copy(), ddepth=cv2.CV_64F, kernel=kernel).astype('int32')
                    S_in_hpf = cv2.filter2D(src=S_in.copy(), ddepth=cv2.CV_64F, kernel=kernel).astype('int32')

                    V_um = V_in - port_list[PORT_L_V_POS] * V_in_hpf
                    V_um = np.clip(V_um, 0, 255).astype('uint8')
                    # V_um = np.clip(V_um, 0, 255)

                    V_ums = V_in - port_list[PORT_L_V_POS] * V_in_hpf - port_list[PORT_L_S_POS] * S_in_hpf * rho
                    V_ums = np.clip(V_ums, 0, 255).astype('uint8')
                    # V_ums = np.clip(V_ums, 0, 255)

                    # Step 7: Calculate weights
                    V_var_V_in = local_variance(V_in, kernel_size=port_list[PORT_T_POS])
                    V_var_V_bf = local_variance(V_bf, kernel_size=port_list[PORT_T_POS])
                    div = np.divide(V_var_V_bf, V_var_V_in, where=V_var_V_in != 0)
                    w = div * (div < 1)

                    # Step 8: Variant
                    # enhanced_V = (w * V_in + (1 - w) * V_um).astype('uint8')
                    enhanced_V = (w * V_um + (1 - w) * V_ums).astype('uint8')

                    # enhanced_V = cv2.normalize(src=enhanced_V, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                    enhanced_V = np.clip(enhanced_V, 0, 255)

                    # Step 8: Merge the enhanced V channel back and convert the image to BGR color space
                    enhanced_hsv = cv2.merge((H_in, S_in, enhanced_V))
                    enhanced_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

                    # Step 0: convert image
                    if len(p_in.arr.shape) == 3:
                        # Convert the BRG image to RGB
                        p_out.arr[:] = enhanced_bgr
                    else:
                        p_out.arr[:] = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

                else:
                    p_out.arr[:] = p_in.arr[:]
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("SUM JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def apply_clipped_histogram(img, clipped_histogram):
    # Compute the cumulative distribution function (CDF) of the clipped histogram
    cdf = np.cumsum(clipped_histogram)
    cdf_normalized = cdf * (clipped_histogram.shape[0] - 1) / cdf[-1]

    # Create a lookup table to map the original pixel values to the equalized pixel values
    lookup_table = np.zeros_like(img, dtype=np.uint8)
    for i in range(clipped_histogram.shape[0]):
        lookup_table[img == i] = cdf_normalized[i]

    return lookup_table


def main_hist_equalization_um(port_list: list = None) -> bool:
    """
    :param port_list: Param needed list of port names [input1,  wave_offset, kernel_size, sigma, output]
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
    PORT_STR_1_POS = 3
    # noinspection PyPep8Naming
    PORT_STR_2_POS = 4
    # noinspection PyPep8Naming
    PORT_OUT_POS = 5

    # check if param OK
    if len(port_list) != 6:
        log_error_to_console("HE-UM JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                if port_list[PORT_KERNEL_POS] == None:
                    kernel = None
                elif 'xy' in port_list[PORT_KERNEL_POS]:
                    kernel = eval('Application.Jobs.kernels.' + port_list[PORT_KERNEL_POS])
                else:
                    kernel = np.array(eval(port_list[PORT_KERNEL_POS]))

                if kernel is not None:
                    in_img = p_in.arr.copy()

                    # Step 0: If the image is RGB transform to YCrBr and work only with luminace
                    if len(p_in.arr.shape) == 3:
                        img_ycbcr = cv2.cvtColor(p_in.arr, cv2.COLOR_BGR2YCR_CB)
                        in_img = img_ycbcr[:, :, 0]

                    # Step 1: Compute UM
                    hpf = cv2.filter2D(src=in_img.copy(), ddepth=cv2.CV_16S, kernel=kernel)
                    um = in_img - port_list[PORT_STR_1_POS] * hpf
                    um = np.clip(um, 0, 255).astype('uint8')

                    # Step 2: Clipping threshold

                    histogram_um = cv2.calcHist([um], [0], None, [256], [0, 256])
                    clipping_thr = np.sum(histogram_um) / len(histogram_um)

                    # Calculate the clipped histogram
                    clipped_histogram = np.where(histogram_um >= clipping_thr, clipping_thr, histogram_um)

                    # Apply the clipped histogram to the image
                    clipped_img = apply_clipped_histogram(um, clipped_histogram)

                    # Step 3: Segmentation threshold
                    segmentation_thr = np.mean(clipped_histogram)

                    F_l = np.where(clipped_img <= segmentation_thr, clipped_img, 0)
                    F_h = np.where(clipped_img > segmentation_thr, clipped_img, 0)

                    # Step 4: Equalization process
                    equalized_F_l = cv2.equalizeHist(F_l)
                    equalized_F_h = cv2.equalizeHist(F_h)

                    # Step 5: Combine F_l and F_h
                    equalized_img = equalized_F_l + equalized_F_h

                    # Step 6: Compute UM
                    hpf = cv2.filter2D(src=equalized_img.copy(), ddepth=cv2.CV_16S, kernel=kernel)
                    img_out = equalized_img - port_list[PORT_STR_2_POS] * hpf
                    img_out = np.clip(img_out, 0, 255).astype('uint8')

                    if len(p_in.arr.shape) == 3:
                        img_ycbcr[:, :, 0] = img_out
                        img_out = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2BGR)

                    p_out.arr[:] = img_out
                else:
                    p_out.arr[:] = p_in.arr[:]
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("HE_UM JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


############################################################################################################################################
# Job create functions
############################################################################################################################################

def do_histogram_equalization_job(port_input_name: str, save_histogram = True,
                                  port_img_output: str = None, is_rgb=False,
                                  level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Equalizes the histogram of a grayscale image. Implementation uses opencv implementation.
    This method usually increases the global contrast of many images, especially when the image is represented by a narrow range of intensity values.
    Through this adjustment, the intensities can be better distributed on the histogram utilizing the full range of intensities evenly.
    This allows for areas of lower local contrast to gain a higher contrast. Histogram equalization accomplishes this by effectively spreading out the highly
    populated intensity values which use to degrade image contrast.
    :param port_input_name: Name of input port
    :param port_img_output: Name of output port
    :param save_histogram: If we desire to save the histogram from this processing
    :param level: Level of input port, please correlate with each input port name parameter
    :param is_rgb: if output port is rgb or not
    :param wave_offset: wave of input port, please correlate with each input port name parameter
    :return: Name of output port or ports
    """
    # Do this for each input port this function has
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_img_output is None:
        port_img_output = 'HIST_EQUAL_{Input}'.format(Input=port_input_name)

    # size can be custom as needed
    port_img_output_name = transform_port_name_lvl(name=port_img_output, lvl=level)
    port_img_output_name_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, port_img_output_name, save_histogram]
    output_port_list = [(port_img_output_name, port_img_output_name_size, 'B', True)]

    job_name = job_name_create(action='HISTOGRAM EQUALIZATION ', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_histogram_equalization',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_img_output


def do_sharpen_filter_job(port_input_name: str, kernel: str,
                          port_output_name: str = None,
                          is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Sharpen filter in image processing improves spatial resolution by enhancing object boundaries but at the cost of image noise.
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: kernel size of sharpen filter
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if kernel is None:
        kernel = None
    elif isinstance(kernel, list):
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel = kernel.__str__()
    else:
        if not isinstance(kernel, str):
            log_setup_info_to_console("HPF FILTER JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    if port_output_name is None:
        port_output_name = 'HPF_' + str(kernel).replace('.', '_') + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='HPF', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               Kernel=str(kernel))

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_sharpen_filter_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_unsharp_filter_expanded_job(port_input_name: str,  kernel: str, strenght: float,
                                   port_output_name: str = None,
                                   wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    The unsharp filter is a simple sharpening operator which derives its name from the fact that it enhances edges
    (and other high frequency components in an image) via a procedure which subtracts an unsharp, or smoothed,
    version of an image from the original image. The unsharp filtering technique is commonly used in the photographic
    and printing industries for crisping edges. The implementation is done using PIL-image library.
    By default the radius is 2 and percent is 150.
    https://pdfs.semanticscholar.org/a9ea/aecf23f3a4b7822e4bcca924e02cd5b4dc4e.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: smoothing kernel to use
    :param strenght: alpha constant that represents the strenght
    :param threshold: threshold to apply
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if kernel is None:
        kernel = None
    elif isinstance(kernel, list):
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel = kernel.__str__()
    else:
        if not isinstance(kernel, str):
            log_setup_info_to_console("UM JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    if port_output_name is None:
        port_output_name = 'UM_' + str(kernel).replace('.', '_') + '_S_' + str(strenght).replace('.', '_') + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel, strenght, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='UM', input_list=input_port_list, wave_offset=[wave_offset], level=level, Kernel=str(kernel), S=str(strenght).replace('.', '_'))

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_unsharp_filter_func_long',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_unsharp_filter_dilated_2dwt_job(port_input_name: str,  kernel: str, strenght: float, levels_fusion: int, wave_lenght: str, fusion_rule: str,
                                       port_output_name: str = None,
                                       wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    xxx
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: smoothing kernel to use
    :param strenght: alpha constant that represents the strenght
    :param threshold: threshold to apply
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if kernel is None:
        kernel = None
    elif isinstance(kernel, list):
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel = kernel.__str__()
    else:
        if not isinstance(kernel, str):
            log_setup_info_to_console("UM_DILATED_2DWT FILTER JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    if port_output_name is None:
        port_output_name = 'UM_DILATED_2DWT_' + str(kernel).replace('.', '_') + '_S_' + str(strenght).replace('.', '_') + '_L_' + \
                           levels_fusion.__str__() + '_' + wave_lenght.upper() + '_F_' + fusion_rule + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel, strenght, levels_fusion, wave_lenght, output_port_name, fusion_rule]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='UM dilated 2DWT', input_list=input_port_list, wave_offset=[wave_offset], level=level, Kernel=str(kernel),
                               S=str(strenght).replace('.', '_'), levels_fusion=levels_fusion)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_um_dilated_2dwt',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_um_2dwt_fusion(port_input_name: str,  octaves: int, m: int, k: float, s: float, wavelet: str,
                                       port_output_name: str = None,
                                       wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    The proposed approach uses a multi-scale scheme and a wavelet based fusion algorithm. Specifically, the input image is initially processed by a cluster
    of un-sharp filters with different variance. The final image is then obtained with the aid of wavelet fusion. The application of the unsharp filters with
    different size Gaussian filters emphasizes important information in different frequency bands. It is shown that the proposed technique can be used as a
    preprocessing stage to general image fusion approaches. The quality of the resulting images is evaluated using three different sharpening indices.
    https://ieeexplore.ieee.org/document/6916146
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param octaves: number of octaves
    :param m: variable that controls the amount of smoothing with typical values in the range [1,2]
    :param k: coefficient to differ between two successive filters
    :param s: Gaussian filters standard deviation
    :param wavelet: wavelet to use ['haar', 'db4', 'bior3.5']
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'UM_2DWT_FUSION_O_' + str(octaves).replace('.', '_') +\
                           '_m_' + str(m) + '_k_' + str(k).replace('.', '_') + '_s_' + str(s).replace('.', '_') + '_' + wavelet.upper() + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, octaves, m, k, s, wavelet, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='UM dilated 2DWT', input_list=input_port_list, wave_offset=[wave_offset], level=level, octaves=str(octaves),
                               m=str(m).replace('.', '_'), k=str(k).replace('.', '_'), s=str(s).replace('.', '_'), wavelet=wavelet)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_um_2dwt_fusion',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_unsharp_filter_job(port_input_name: str,
                          radius: int = 2, percent: int = 150, threshold=3, port_output_name: str = None,
                          wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    The unsharp filter is a simple sharpening operator which derives its name from the fact that it enhances edges
    (and other high frequency components in an image) via a procedure which subtracts an unsharp, or smoothed,
    version of an image from the original image. The unsharp filtering technique is commonly used in the photographic
    and printing industries for crisping edges. The implementation is done using PIL-image library.
    By default the radius is 2 and percent is 150.
    https://pdfs.semanticscholar.org/a9ea/aecf23f3a4b7822e4bcca924e02cd5b4dc4e.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param radius: radius of filter
    :param percent: percent of dark to add
    :param threshold: threshold to apply
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'UNSHARP_FILER_R_' + str(radius) + '_P_' + str(percent) + '_T_' + str(threshold) + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, radius, percent, threshold, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Unsharp filter', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               R=str(radius), P=str(percent))

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_unsharp_filter_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_constrained_unsharp_filter_expanded_job(port_input_name: str,
                                               laplace_kernel: str, hpf_strenght: float,
                                               lee_sigma_filter_window: int, lee_filter_sigma_value:int,
                                               threshold_cliping_window:int,
                                               casacade_version: float = False,
                                               port_output_name: str = None,
                                               wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    TBD
    https://link.springer.com/chapter/10.1007/978-3-540-69905-7_2
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: smoothing kernel to use
    :param strenght: alpha constant that represents the strenght
    :param threshold: threshold to apply
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if laplace_kernel is None:
        laplace_kernel = None
    elif isinstance(laplace_kernel, list):
        if laplace_kernel not in custom_kernels_used:
            custom_kernels_used.append(laplace_kernel)
        kernel = laplace_kernel.__str__()
    else:
        if not isinstance(laplace_kernel, str):
            log_setup_info_to_console("CONSTRAINED UNSHARP MASKING JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            laplace_kernel = laplace_kernel.lower() + '_xy'

    if port_output_name is None:
        if casacade_version:
            port_output_name = 'C_CUM_' + str(laplace_kernel).replace('.', '_') + '_S_' + str(hpf_strenght).replace('.', '_') \
                               + '_LS_W_' + str(lee_sigma_filter_window) + '_LS_V_' + str(lee_filter_sigma_value) \
                               + '_TH_' + str(threshold_cliping_window) + '_' + port_input_name
        else:
            port_output_name = 'CUM_' + str(laplace_kernel).replace('.', '_') + '_S_' + str(hpf_strenght).replace('.', '_') \
                               + '_LS_W_' + str(lee_sigma_filter_window) + '_LS_V_' + str(lee_filter_sigma_value)\
                               + '_TH_' + str(threshold_cliping_window) + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, laplace_kernel, hpf_strenght, lee_sigma_filter_window, lee_filter_sigma_value, threshold_cliping_window, casacade_version, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    if casacade_version:
        job_name = job_name_create(action='CUM', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                                   Kernel=str(laplace_kernel), HPF_S=str(hpf_strenght).replace('.', '_'), W=str(lee_sigma_filter_window), S=str(lee_filter_sigma_value),
                                   TH=str(threshold_cliping_window))
    else:
        job_name = job_name_create(action='C-CUM', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                                   Kernel=str(laplace_kernel), HPF_S=str(hpf_strenght).replace('.', '_'), W=str(lee_sigma_filter_window), S=str(lee_filter_sigma_value),
                                   TH=str(threshold_cliping_window))

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_constrained_unsharp_filter',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_nonlinear_unsharp_filter_job(port_input_name: str,
                                    kernel_size: int, sigma: float,
                                    operator,
                                    laplace_kernel: str, hpf_strength: float,
                                    port_output_name: str = None,
                                    wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    TBD
    https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-5/issue-3/0000/Nonlinear-unsharp-masking-methods-for-image-contrast-enhancement/10.1117/12.242618.short
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param sigma: sigma to use. 0 to auto calculate
    :param kernel_size: kernel of gaussian to use
    :param operator: operator to use
    :param laplace_kernel: xxx
    :param hpf_strength: xxx
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    kernel_x = operator.lower() + '_x'
    kernel_y = operator.lower() + '_y'

    kernel_1 = kernel_x
    kernel_2 = kernel_y

    # check kernel passed
    if isinstance(kernel_x, list):
        if kernel_1 not in custom_kernels_used:
            custom_kernels_used.append(kernel_1)
        kernel_1 = kernel_1.__str__()
    else:
        if not isinstance(kernel_x, str):
            log_setup_info_to_console("CONVOLUTION JOB DIDN'T RECEIVE CORRECT KERNEL")
            return None

    if isinstance(kernel_y, list):
        if kernel_2 not in custom_kernels_used:
            custom_kernels_used.append(kernel_2)
        kernel_2 = kernel_2.__str__()
    else:
        if not isinstance(kernel_y, str):
            log_setup_info_to_console("CONVOLUTION JOB DIDN'T RECEIVE CORRECT KERNEL")
            return None

    if laplace_kernel is None:
        laplace_kernel = None
    elif isinstance(laplace_kernel, list):
        if laplace_kernel not in custom_kernels_used:
            custom_kernels_used.append(laplace_kernel)
        kernel = laplace_kernel.__str__()
    else:
        if not isinstance(laplace_kernel, str):
            log_setup_info_to_console("NONLINEAR UNSHARP MASKING JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            laplace_kernel = laplace_kernel.lower() + '_xy'

    if port_output_name is None:
        # port_output_name = 'NUM_' + + '_' + port_input_name
        port_output_name = 'NUM_GAUSS_BLUR_K_{}_S_{}_{}_{}_Str_{}_{}'.format(kernel_size, str(sigma).replace('.', '_'), operator,
                                                                             str(laplace_kernel).replace('.', '_'), str(hpf_strength).replace('.', '_'), port_input_name)

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel_size, sigma, kernel_1, kernel_2, laplace_kernel, hpf_strength, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='NUM', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               GAUSS_S=sigma.__str__().replace('.','_'), GAUSS_K=kernel_size, OPERATOR=operator,
                               LAPLACE_K=str(laplace_kernel), HPF_S=str(hpf_strength).replace('.', '_'),)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_nonlinear_unsharp_filter',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_normalized_unsharp_filter_job(port_input_name: str,  kernel: str, strenght: float,
                                     port_output_name: str = None,
                                     wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    https://www.researchgate.net/profile/Giovanni-gianni-Ramponi/publication/220050577_Nonlinear_unsharp_masking_methods_for_image_contrast_enhancement/links/542192f90cf203f155c6e2eb/Nonlinear-unsharp-masking-methods-for-image-contrast-enhancement.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: smoothing kernel to use
    :param strenght: alpha constant that represents the strenght
    :param threshold: threshold to apply
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if kernel is None:
        kernel = None
    elif isinstance(kernel, list):
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel = kernel.__str__()
    else:
        if not isinstance(kernel, str):
            log_setup_info_to_console("N-NUM JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    if port_output_name is None:
        port_output_name = 'N_NUM_' + str(kernel).replace('.', '_') + '_S_' + str(strenght).replace('.', '_') + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel, strenght, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='N_NUM', input_list=input_port_list, wave_offset=[wave_offset], level=level, Kernel=str(kernel), S=str(strenght).replace('.', '_'))

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  # main_func_name='main_unsharp_filter_func_long',
                                  main_func_name='main_n_nun_func_long',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_adaptive_non_linear_unsharp_filter_job(port_input_name: str,  kernel: str,
                                             thr_1: int, thr_2: int,
                                             bf_distance: int = 9, bf_sigma_colors: int = 75, bf_sigma_space: int = 75,
                                             port_output_name: str = None,
                                             wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    https://www.sciencedirect.com/science/article/abs/pii/S0165168498000425
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: smoothing kernel to use
    :param strenght: alpha constant that represents the strenght
    :param threshold: threshold to apply
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if kernel is None:
        kernel = None
    elif isinstance(kernel, list):
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel = kernel.__str__()
    else:
        if not isinstance(kernel, str):
            log_setup_info_to_console("ANUM JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    if port_output_name is None:
        port_output_name = 'ANUM_' + str(kernel).replace('.', '_') + '_BF_D_' + str(bf_distance) + '_SC_' + str(bf_sigma_colors).replace('.', '_') + '_SS_' + str(bf_sigma_space).replace('.', '_') \
                           + '_THR_1_' + str(thr_1) + '_THR_1_' + str(thr_2) + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel, bf_distance, bf_sigma_colors, bf_sigma_colors, thr_1, thr_2, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='ANUM', input_list=input_port_list, wave_offset=[wave_offset], level=level, Kernel=str(kernel),
                               BF_D=str(bf_distance).replace('.', '_'), BF_SC=str(bf_sigma_colors).replace('.', '_'), BF_SS=str(bf_sigma_space).replace('.', '_'),
                               THR_1=str(thr_1).replace('.', '_'), THR_2=str(thr_2).replace('.', '_'))

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  # main_func_name='main_unsharp_filter_func_long',
                                  main_func_name='main_adaptive_num_func_long',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_adaptive_unsharp_filter_job(port_input_name: str,  kernel: str,
                                   thr_1: int = 60, thr_2: int = 200, alpha_low: int = 4, alpha_high: int = 3,
                                   mu: float = 0.1, beta: float = 0.5,
                                   port_output_name: str = None,
                                   wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    https://ieeexplore.ieee.org/abstract/document/826787?casa_token=Fy16x0iN_PMAAAAA:_AkzDUU-gcWkVPc2jxOBNOhG80UWHjf-jnpvWTpJb29MLz_FsmjIpKRHqYvSezZ-tXGKgRjLPg
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: smoothing kernel to use
    :param thr_1: low threshold for variance
    :param thr_2: low threshold for variance
    :param alpha_low: gain for low variance
    :param alpha_high: gain for high variance
    :param mu: parameter in lambda gain adaptation
    :param beta: parameter in autocorrection matrix
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if kernel is None:
        kernel = None
    elif isinstance(kernel, list):
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel = kernel.__str__()
    else:
        if not isinstance(kernel, str):
            log_setup_info_to_console("AUM JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    if port_output_name is None:
        port_output_name = 'AUM_' + str(kernel).replace('.', '_') + '_T1_' + str(thr_1) + '_T2_' + str(thr_2)\
                           + '_AL_' + str(alpha_low) + '_AH_' + str(alpha_high)\
                           + '_MU_' + str(mu).replace('.', '_') + '_BETA_' + str(beta).replace('.', '_')  + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel, thr_1, thr_2, alpha_low, alpha_high, mu, beta, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='AUM', input_list=input_port_list, wave_offset=[wave_offset], level=level, Kernel=str(kernel),
                               T1=str(thr_1), T2=str(thr_2), AL=str(alpha_low), AH=str(alpha_high),
                               MU=str(mu).replace('.', '_'), BETA=str(beta).replace('.', '_'))

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_adaptive_um',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_selective_unsharp_filter_job(port_input_name: str,  kernel: str,
                                    lambda_v: float, lambda_s: float, T: int =3,
                                    bf_distance: int = 9, bf_sigma_colors: int = 75, bf_sigma_space: int = 75,
                                    port_output_name: str = None,
                                    wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    http://www.ijeee.net/uploadfile/2013/0510/20130510113835907.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: smoothing kernel to use
xxxx
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if kernel is None:
        kernel = None
    elif isinstance(kernel, list):
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel = kernel.__str__()
    else:
        if not isinstance(kernel, str):
            log_setup_info_to_console("SUM JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    if port_output_name is None:
        port_output_name = 'SUM_' + str(kernel).replace('.', '_') + '_BF_D_' + str(bf_distance) + '_SC_' + str(bf_sigma_colors).replace('.', '_') + '_SS_' + str(bf_sigma_space).replace('.', '_') \
                           + '_STR_V_' + str(lambda_v).replace('.', '_') + '_STR_S_' + str(lambda_s).replace('.', '_') + '_T_' + str(T) +'_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel, bf_distance, bf_sigma_colors, bf_sigma_colors, lambda_v, lambda_s, T, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='SUM', input_list=input_port_list, wave_offset=[wave_offset], level=level, Kernel=str(kernel),
                               BF_D=str(bf_distance).replace('.', '_'), BF_SC=str(bf_sigma_colors).replace('.', '_'), BF_SS=str(bf_sigma_space).replace('.', '_'),
                               STR_V=str(lambda_v).replace('.', '_'), STR_S=str(lambda_s).replace('.', '_'), T=str(T))

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_selective_um',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_histogram_equalization_unsharp_filter_job(port_input_name: str,  kernel: str,
                                                 strength_1: float, strength_2: float,
                                                 port_output_name: str = None,
                                                 wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    http://www.ijeee.net/uploadfile/2013/0510/20130510113835907.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: smoothing kernel to use
    :param strength_1: strength of first UM
    :param strength_2: strength of second UM
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if kernel is None:
        kernel = None
    elif isinstance(kernel, list):
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel = kernel.__str__()
    else:
        if not isinstance(kernel, str):
            log_setup_info_to_console("HE_UM JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    if port_output_name is None:
        port_output_name = 'HE_UM_' + str(kernel).replace('.', '_') + '_STR_1_' + str(strength_1).replace('.', '_') + '_STR_2_' + str(strength_2).replace('.', '_') + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel, strength_1, strength_2, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='HE-UM', input_list=input_port_list, wave_offset=[wave_offset], level=level, Kernel=str(kernel),
                               STR_1=str(strength_1).replace('.', '_'), STR_2=str(strength_2).replace('.', '_'))

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_hist_equalization_um',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name
if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
