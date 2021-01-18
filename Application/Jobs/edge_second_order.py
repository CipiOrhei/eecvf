# noinspection PyPackageRequirements
import cv2
import numpy as np
from numba import jit

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.port import Port
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console
# Do not delete used indirectly
# noinspection PyUnresolvedReferences
import Application.Jobs.kernels


def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """

    return JobInitStateReturn(True)


def main_func_laplacian_pyramid_2_images(param_list: list = None) -> bool:
    """
    Main function for calculating Laplacian pyramid from one pyramid level and
    expansion of the next one
    :param param_list: Param needed to respect the following list:
                       [port_in_name_high_level: str, port_in_name_expanded_level: str, wave_offset,
                        port_out_name name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG = 0
    # noinspection PyPep8Naming
    PORT_IN_IMG_2 = 1
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3

    if len(param_list) != 4:
        log_error_to_console("LAPLACIAN PYRAMID IMG DIF JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in_img_1 = get_port_from_wave(name=param_list[PORT_IN_IMG], wave_offset=param_list[PORT_IN_WAVE])
        port_in_img_2 = get_port_from_wave(name=param_list[PORT_IN_IMG_2], wave_offset=param_list[PORT_IN_WAVE])
        port_out_img = get_port_from_wave(name=param_list[PORT_OUT_POS])

        if (port_in_img_1.is_valid() and port_in_img_2.is_valid()) is True:
            try:
                port_out_img.arr[:] = cv2.subtract(src1=port_in_img_1.arr, src2=port_in_img_2.arr)
                port_out_img.set_valid()
            except BaseException as error:
                log_error_to_console("LAPLACIAN PYRAMID IMG DIF JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_func_binary_laplace_2_images(param_list: list = None) -> bool:
    """
    Main function for calculating binary Laplacian image.
    The resulting image B = Smooth-Image is the band-limited Laplacian of the image. F
    :param param_list: Param needed to respect the following list:
                       [port_in_name_high_level: str, port_in_name_expanded_level: str, wave_offset,
                        port_out_name name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG = 0
    # noinspection PyPep8Naming
    PORT_IN_IMG_2 = 1
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3
    # noinspection PyPep8Naming
    PORT_BINARY_POS = 4

    if len(param_list) != 5:
        log_error_to_console("LAPLACIAN BINARY IMG DIF JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in_img_1 = get_port_from_wave(name=param_list[PORT_IN_IMG], wave_offset=param_list[PORT_IN_WAVE])
        port_in_img_2 = get_port_from_wave(name=param_list[PORT_IN_IMG_2], wave_offset=param_list[PORT_IN_WAVE])
        port_out_img = get_port_from_wave(name=param_list[PORT_OUT_POS])

        if (port_in_img_1.is_valid() and port_in_img_2.is_valid()) is True:
            try:
                port_out_img.arr[:] = port_in_img_2.arr.astype(np.dtype('h')) - port_in_img_1.arr.astype(np.dtype('h'))

                if param_list[PORT_BINARY_POS]:
                    port_out_img.arr[:] = port_out_img.arr > 0.0
                    port_out_img.arr[:] *= 255
                port_out_img.set_valid()
            except BaseException as error:
                log_error_to_console("LAPLACIAN BINARY JOB NOK: ", str(error.args[0]))
                pass
        else:
            return False

        return True


def main_func_laplace(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job for second order derivative jobs.
    :param param_list: Param needed to respect the following list:
                       [port_in name: str ,  wave_offset,
                        port_kernel: what kernel do you want to use
                        port_out_name name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_INPUT = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_THR = 2
    # noinspection PyPep8Naming
    PORT_KERNEL = 3
    # noinspection PyPep8Naming
    PORT_OUT_POS = 4

    # verify that the number of parameters are OK.
    if len(param_list) != 5:
        log_error_to_console("LAPLACE OPERATOR JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_INPUT], wave_offset=param_list[PORT_IN_WAVE])
        port_out = get_port_from_wave(name=param_list[PORT_OUT_POS])

        if 'xy' in param_list[PORT_KERNEL]:
            kernel = eval('Application.Jobs.kernels.' + param_list[PORT_KERNEL])
        else:
            kernel = np.array(eval(param_list[PORT_KERNEL]))
        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                result = cv2.filter2D(src=port_in.arr.copy(), ddepth=cv2.CV_16SC1, kernel=kernel)
                result = np.uint16(np.absolute(result))
                if param_list[PORT_IN_THR] != 0:
                    retval, result = cv2.threshold(src=result, thresh=param_list[PORT_IN_THR], maxval=255, type=cv2.NORM_MINMAX)

                port_out.arr[:] = result
                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("LAPLACE OPERATOR JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


@jit(nopython=True)
def zero_crossing_calc(port_in: Port.arr, threshold: int):
    """
    Function for zero crossing
    :param port_in: image input
    :param threshold: threshold for the zero crossing
    :return: True if the job executed OK.
    """

    zero_crossing = np.zeros(shape=port_in.shape)

    sign = port_in > 0
    # noinspection PyPep8Naming
    T = threshold * max(abs(port_in.max()), abs(port_in.min()))

    # computing zero crossing
    for i in range(1, zero_crossing.shape[0] - 1):
        for j in range(1, zero_crossing.shape[1] - 1):
            if (sign[i - 1][j - 1] is not sign[i + 1][j + 1] and abs(port_in[i - 1][j - 1] - port_in[i + 1][j + 1]) > T) or \
                    (sign[i - 1][j + 0] is not sign[i + 1][j - 0] and abs(port_in[i - 1][j + 0] - port_in[i + 1][j - 0]) > T) or \
                    (sign[i - 1][j + 1] is not sign[i + 1][j - 1] and abs(port_in[i - 1][j + 1] - port_in[i + 1][j - 1]) > T) or \
                    (sign[i - 0][j - 1] is not sign[i + 0][j + 1] and abs(port_in[i - 0][j + 1] - port_in[i + 0][j - 1]) > T):
                zero_crossing[i][j] = 255

    return zero_crossing


def main_func_zero_crossing(param_list: list = None) -> bool:
    """
    Main function for zero crossing function. Algorithm for detecting local maxima's and minima's in a signal.
    :param param_list: Param needed to respect the following list:
                       [port_in name: str,    wave_offset,
                        port_kernel: what kernel do you want to use
                        port_out_name name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_INPUT = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_THR = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3

    # verify that the number of parameters are OK.
    if len(param_list) != 4:
        log_error_to_console("ZERO CROSSING JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_INPUT], wave_offset=param_list[PORT_IN_WAVE])
        port_out = get_port_from_wave(name=param_list[PORT_OUT_POS])
        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                if len(port_in.arr.shape) == 3:
                    t = zero_crossing_calc(port_in=port_in.arr[:, :, 0], threshold=param_list[PORT_IN_THR]) + \
                        zero_crossing_calc(port_in=port_in.arr[:, :, 1], threshold=param_list[PORT_IN_THR]) + \
                        zero_crossing_calc(port_in=port_in.arr[:, :, 2], threshold=param_list[PORT_IN_THR])
                    result, port_out.arr[:] = cv2.threshold(src=t, thresh=255, maxval=255, type=cv2.NORM_MINMAX)
                else:
                    port_out.arr[:] = zero_crossing_calc(port_in=port_in.arr, threshold=param_list[PORT_IN_THR])
                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("ZERO CROSSING JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_func_marr_hildreth(param_list: list = None) -> bool:
    """
    http://www.hms.harvard.edu/bss/neuro/bornlab/qmbc/beta/day4/marr-hildreth-edge-prsl1980.pdf
    Main function for gradient calculation job for second order derivative jobs.
    :param param_list: Param needed to respect the following list:
                       [port_in name: str
                        port_kernel: what kernel do you want to use
                        port_out_name name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_INPUT = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_BLUR_SIZE = 2
    # noinspection PyPep8Naming
    PORT_KERNEL_BLUR_SIGMA = 3
    # noinspection PyPep8Naming
    PORT_KERNEL_LAPLACE_SIGMA = 4
    # noinspection PyPep8Naming
    PORT_PRECALCULATED_LAPLACE = 5
    # noinspection PyPep8Naming
    PORT_ZC_PARAM = 6
    # noinspection PyPep8Naming
    PORT_OUT_POS = 7

    # verify that the number of parameters are OK.
    if len(param_list) != 8:
        log_error_to_console("MARR HILDRETH OPERATOR JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_INPUT], wave_offset=param_list[PORT_IN_WAVE])
        port_out = get_port_from_wave(name=param_list[PORT_OUT_POS])

        if 'xy' in param_list[PORT_KERNEL_LAPLACE_SIGMA]:
            kernel = eval('Application.Jobs.kernels.' + param_list[PORT_KERNEL_LAPLACE_SIGMA])
        else:
            kernel = np.array(eval(param_list[PORT_KERNEL_LAPLACE_SIGMA]))
        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                gaus = np.zeros_like(port_in.arr)

                if param_list[PORT_PRECALCULATED_LAPLACE] is False:
                    gaus = cv2.GaussianBlur(src=port_in.arr.copy(),
                                            ksize=(param_list[PORT_KERNEL_BLUR_SIZE], param_list[PORT_KERNEL_BLUR_SIZE]),
                                            sigmaX=param_list[PORT_KERNEL_BLUR_SIGMA])

                log = cv2.filter2D(src=gaus, ddepth=cv2.CV_16SC1, kernel=kernel)

                if len(log.shape) == 3:
                    t = zero_crossing_calc(port_in=log[:, :, 0], threshold=param_list[PORT_ZC_PARAM]) + \
                        zero_crossing_calc(port_in=log[:, :, 1], threshold=param_list[PORT_ZC_PARAM]) + \
                        zero_crossing_calc(port_in=log[:, :, 2], threshold=param_list[PORT_ZC_PARAM])
                    result, port_out.arr[:] = cv2.threshold(src=t, thresh=255, maxval=255, type=cv2.NORM_MINMAX)
                else:
                    port_out.arr[:] = zero_crossing_calc(port_in=log, threshold=param_list[PORT_ZC_PARAM])

                # port_out.arr[:] = zero_crossing_calc(port_in=log, threshold=param_list[PORT_ZC_PARAM])

                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("MARR HILDRETH JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


if __name__ == "__main__":
    pass
