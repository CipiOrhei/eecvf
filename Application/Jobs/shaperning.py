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
                    result = cv2.filter2D(p_in.arr.copy(), -1, kernel_high_pass)
                        # p_out.arr[:] = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
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

                    lap = cv2.filter2D(src=p_in.arr.copy(), ddepth=cv2.CV_64F, kernel=kernel)
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

from numba import jit

@jit(nopython=True)
def fusion_test(coef_list, octavs, fuison_logic):
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

            # if port_list[PORT_FUSION_POS] == 'mean':
            #     LL[i][j] = np.mean(l_ll)
            #     LH[i][j] = np.mean(l_lh)
            #     HL[i][j] = np.mean(l_hl)
            #     HH[i][j] = np.mean(l_hh)
            if fuison_logic == 'average':
                LL[i][j] = sum(l_ll)/octavs
                LH[i][j] = sum(l_lh)/octavs
                HL[i][j] = sum(l_hl)/octavs
                HH[i][j] = sum(l_hh)/octavs
            elif fuison_logic == 'max':
                LL[i][j] = max(l_ll)
                LH[i][j] = max(l_lh)
                HL[i][j] = max(l_hl)
                HH[i][j] = max(l_hh)
            elif fuison_logic == 'min':
                LL[i][j] = min(l_ll)
                LH[i][j] = min(l_lh)
                HL[i][j] = min(l_hl)
                HH[i][j] = min(l_hh)

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

                    for l in range(port_list[PORT_FUSION_LVL]):
                        res_um = um(img=p_in.arr, kernel=dilate(kernel, l), strength=port_list[PORT_STREGHT_POS])
                        coef_list.append(pywt.dwt2(res_um.astype('int32'), port_list[PORT_WAVELENGT]))

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

                            for k in range(port_list[PORT_FUSION_LVL]):
                                l_ll.append(coef_list[k][0][i][j])
                                l_lh.append(coef_list[k][1][0][i][j])
                                l_hl.append(coef_list[k][1][1][i][j])
                                l_hh.append(coef_list[k][1][2][i][j])

                            # if port_list[PORT_FUSION_POS] == 'mean':
                            #     LL[i][j] = np.mean(l_ll)
                            #     LH[i][j] = np.mean(l_lh)
                            #     HL[i][j] = np.mean(l_hl)
                            #     HH[i][j] = np.mean(l_hh)
                            if port_list[PORT_FUSION_POS] == 'average':
                                LL[i][j] = np.average(l_ll)
                                LH[i][j] = np.average(l_lh)
                                HL[i][j] = np.average(l_hl)
                                HH[i][j] = np.average(l_hh)
                            elif port_list[PORT_FUSION_POS] == 'max':
                                LL[i][j] = max(l_ll)
                                LH[i][j] = max(l_lh)
                                HL[i][j] = max(l_hl)
                                HH[i][j] = max(l_hh)
                            elif port_list[PORT_FUSION_POS] == 'min':
                                LL[i][j] = min(l_ll)
                                LH[i][j] = min(l_lh)
                                HL[i][j] = min(l_hl)
                                HH[i][j] = min(l_hh)

                    coeffs = LL, (LH, HL, HH)

                    # coeffs = fusion_test(coef_list=coef_list, octavs=port_list[PORT_FUSION_LVL], fuison_logic=port_list[PORT_FUSION_POS])

                    inverse = pywt.idwt2(coeffs, port_list[PORT_WAVELENGT])

                    inverse = cv2.normalize(src=inverse, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

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
                for octav in range(port_list[PORT_O_POS]):
                    arg = port_list[PORT_S_LVL] * port_list[PORT_K_LVL] ** octav
                    img_blur = cv2.GaussianBlur(src=p_in.arr.copy(), ksize=(0, 0), sigmaX=arg)

                    res_um = (port_list[PORT_M_POS] + 1) * p_in.arr.copy().astype('int32') - port_list[PORT_M_POS] * img_blur.astype('int32')

                    coef_list.append(pywt.dwt2(res_um, port_list[PORT_WAVELET]))

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

                        for k in range(port_list[PORT_O_POS]):
                            l_ll.append(coef_list[k][0][i][j])
                            l_lh.append(coef_list[k][1][0][i][j])
                            l_hl.append(coef_list[k][1][1][i][j])
                            l_hh.append(coef_list[k][1][2][i][j])

                        LL[i][j] = max(l_ll)
                        LH[i][j] = max(l_lh)
                        HL[i][j] = max(l_hl)
                        HH[i][j] = max(l_hh)

                coeffs = LL, (LH, HL, HH)
                inverse = pywt.idwt2(coeffs, port_list[PORT_WAVELET])

                inverse = cv2.normalize(src=inverse, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

                p_out.arr[:] = inverse

                p_out.set_valid()
            # except BaseException as error:
            #     log_error_to_console("UM DILATED 2DWT JOB NOK: ", str(error))
            #     pass
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
            log_setup_info_to_console("SHARPEN FILTER JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    if port_output_name is None:
        port_output_name = 'SHARPEN_' + str(kernel).replace('.', '_') + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Sharpen filter', input_list=input_port_list, wave_offset=[wave_offset], level=level,
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
            log_setup_info_to_console("SHARPEN FILTER JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    if port_output_name is None:
        port_output_name = 'UNSHARP_FILTER_' + str(kernel).replace('.', '_') + '_S_' + str(strenght).replace('.', '_') + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel, strenght, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Unsharp filter', input_list=input_port_list, wave_offset=[wave_offset], level=level, Kernel=str(kernel), S=str(strenght).replace('.', '_'))

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
                           '_m_' + str(m) + '_k_' + str(k).replace('.', '_') + '_s_' + str(s) + '_' + wavelet.upper() + '_' + port_input_name

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


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
