# noinspection PyPackageRequirements
import cv2
import numpy as np

# noinspection PyUnresolvedReferences
import Application.Jobs.kernels
from Application.Frame import transferJobPorts
from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from scipy.ndimage.filters import convolve
from cv2.ximgproc import GradientDericheX, GradientDericheY

from Application.Utils.misc import rotate_around_center
from Utils.log_handler import log_error_to_console


"""
Module handles kernel convolution image jobs for the APPL block.
"""


def compute_gradients_frei_chen(port_in: transferJobPorts.Port, dilation_factor: int,
                                port_out_name_g1: str, port_out_name_g2: str, port_out_name_g3: str,
                                port_out_name_g4: str, port_out_name_g5: str, port_out_name_g6: str,
                                port_out_name_g7: str, port_out_name_g8: str, port_out_name_g9: str) -> None:
    """
    Adds to specific output ports the result of convolution the image with 9 frei-chen masks
    The function uses cv2.filter2D function
    :param port_in: image to apply convolution on
    :param dilation_factor: dilation_factor of kernel
    :param port_out_name_g1: image resulted after convolution with kernel g1 -> isotropic gradient
    :param port_out_name_g2: image resulted after convolution with kernel g2 -> isotropic gradient
    :param port_out_name_g3: image resulted after convolution with kernel g3 -> ripple
    :param port_out_name_g4: image resulted after convolution with kernel g4 -> ripple
    :param port_out_name_g5: image resulted after convolution with kernel g5 -> line
    :param port_out_name_g6: image resulted after convolution with kernel g6 -> line
    :param port_out_name_g7: image resulted after convolution with kernel g7 -> discrete laplacian
    :param port_out_name_g8: image resulted after convolution with kernel g8 -> discrete laplacian
    :param port_out_name_g9: image resulted after convolution with kernel g9 -> average
    :return: None
    """
    port_out = [get_port_from_wave(name=port_out_name_g1), get_port_from_wave(name=port_out_name_g2),
                get_port_from_wave(name=port_out_name_g3), get_port_from_wave(name=port_out_name_g4),
                get_port_from_wave(name=port_out_name_g5), get_port_from_wave(name=port_out_name_g6),
                get_port_from_wave(name=port_out_name_g7), get_port_from_wave(name=port_out_name_g8),
                get_port_from_wave(name=port_out_name_g9)]

    for kernel in range(len(port_out)):
        result = np.zeros(shape=port_in.arr.shape, dtype=np.float32)

        if dilation_factor == 0:
            kernel_to_use = eval('Application.Jobs.kernels.' + 'frei_chen_v' + str(kernel + 1))
        elif dilation_factor == 1:
            kernel_to_use = eval('Application.Jobs.kernels.' + 'frei_chen_dilated_5x5_v' + str(kernel + 1))
        elif dilation_factor == 2:
            kernel_to_use = eval('Application.Jobs.kernels.' + 'frei_chen_dilated_7x7_v' + str(kernel + 1))
        # flip kernels for a real convolution to be done by cv2.filter2D
        kernel_to_use = kernel_to_use[::-1, ::-1]
        cv2.filter2D(src=port_in.arr, ddepth=cv2.CV_32F, kernel=kernel_to_use, dst=result, anchor=(-1, -1))
        result = result * result

        port_out[kernel].arr[:] = np.int32(result)
        port_out[kernel].set_valid()


def compute_gradients_navatia_babu(port_in: transferJobPorts.Port,
                                   port_out_name_g1: str, port_out_name_g2: str, port_out_name_g3: str,
                                   port_out_name_g4: str, port_out_name_g5: str, port_out_name_g6: str) -> None:
    """
    Adds to specific output ports the result of convolution the image with 9 frei-chen masks
    The function uses cv2.filter2D function
    :param port_in: image to apply convolution on
    :param port_out_name_g1: image resulted after convolution with kernel g1
    :param port_out_name_g2: image resulted after convolution with kernel g2
    :param port_out_name_g3: image resulted after convolution with kernel g3
    :param port_out_name_g4: image resulted after convolution with kernel g4
    :param port_out_name_g5: image resulted after convolution with kernel g5
    :param port_out_name_g6: image resulted after convolution with kernel g6
    :return: None
    """
    port_out = [get_port_from_wave(name=port_out_name_g1), get_port_from_wave(name=port_out_name_g2),
                get_port_from_wave(name=port_out_name_g3), get_port_from_wave(name=port_out_name_g4),
                get_port_from_wave(name=port_out_name_g5), get_port_from_wave(name=port_out_name_g6)]

    for kernel in range(len(port_out)):
        result = np.zeros(port_in.arr.shape, np.float32)
        kernel_to_use = eval('Application.Jobs.kernels.' + 'navatia_babu_5x5_g' + str(kernel + 1))
        # flip kernels for a real convolution to be done by cv2.filter2D
        kernel_to_use = kernel_to_use[::-1, ::-1]
        cv2.filter2D(src=port_in.arr, ddepth=cv2.CV_32F, kernel=kernel_to_use, dst=result, anchor=(-1, -1))

        port_out[kernel].arr[:] = np.int32(result)
        port_out[kernel].set_valid()


def compute_gradients_8_directions(port_in: transferJobPorts.Port,
                                   port_out_name_gn: str, port_out_name_gnw: str, port_out_name_gw: str, port_out_name_gsw: str,
                                   port_out_name_gs: str, port_out_name_gse: str, port_out_name_ge: str, port_out_name_gne: str,
                                   kernel: bytearray) -> None:
    """
    Adds to specific output ports the result of convolution the image with 8 kernels directions: N, NW, W, SW, S, SE, E, and NE
    The function uses cv2.filter2D function
    :param port_in: image to apply convolution on
    :param port_out_name_gn: image resulted after convolution kernel for N direction and picture
    :param port_out_name_gnw: image resulted after convolution kernel for NW direction and picture
    :param port_out_name_gw: image resulted after convolution kernel for W direction and picture
    :param port_out_name_gsw: image resulted after convolution kernel for SW direction and picture
    :param port_out_name_gs: image resulted after convolution kernel for S direction and picture
    :param port_out_name_gse: image resulted after convolution kernel for SE direction and picture
    :param port_out_name_ge: image resulted after convolution kernel for E direction and picture
    :param port_out_name_gne: image resulted after convolution kernel for NE direction and picture
    :param kernel: kernel to use
    :return: None
    """
    port_out = [get_port_from_wave(name=port_out_name_gn), get_port_from_wave(name=port_out_name_gnw),
                get_port_from_wave(name=port_out_name_gw), get_port_from_wave(name=port_out_name_gsw),
                get_port_from_wave(name=port_out_name_gs), get_port_from_wave(name=port_out_name_gse),
                get_port_from_wave(name=port_out_name_ge), get_port_from_wave(name=port_out_name_gne)]

    kernel_to_use = kernel

    for direction in range(len(port_out)):
        result = np.zeros(port_in.arr.shape, np.float32)
        # flip kernels for a real convolution to be done by cv2.filter2D
        kernel_to_use = kernel_to_use[::-1, ::-1]
        cv2.filter2D(src=port_in.arr, ddepth=cv2.CV_32F, kernel=kernel_to_use, dst=result, anchor=(-1, -1))
        port_out[direction].arr[:] = np.int32(result)
        port_out[direction].set_valid()
        kernel_to_use = rotate_around_center(mat=kernel_to_use)


def compute_gradient_filter_2d(port_in: transferJobPorts.Port, port_out_name_gx: str, port_out_name_gy: str,
                               kernel_x: bytearray, kernel_y: bytearray) -> None:
    """
    Adds to specific output ports the result of convolution the image with 2 kernels(x and y)
    The function uses cv2.filter2D function
    :param port_in: image to apply convolution on
    :param port_out_name_gx: image resulted after convolution kernel for x and picture
    :param port_out_name_gy: image resulted after convolution kernel for y and picture
    :param kernel_x: kernel for x direction
    :param kernel_y: kernel for y direction
    :return: None
    """
    port_out_gx = get_port_from_wave(name=port_out_name_gx)
    port_out_gy = get_port_from_wave(name=port_out_name_gy)

    # Magnitude matrices for Ix/dx and Iy/dy
    magnitude_x = np.zeros(shape=port_in.arr.shape, dtype=np.float32)
    magnitude_y = np.zeros(shape=port_in.arr.shape, dtype=np.float32)
    # flip kernels for a real convolution to be done by cv2.filter2D
    kernel_x = kernel_x[::-1, ::-1]
    kernel_y = kernel_y[::-1, ::-1]

    cv2.filter2D(src=port_in.arr, ddepth=cv2.CV_32F, kernel=kernel_x, dst=magnitude_x, anchor=(-1, -1))
    cv2.filter2D(src=port_in.arr, ddepth=cv2.CV_32F, kernel=kernel_y, dst=magnitude_y, anchor=(-1, -1))

    # Convert to signed 16 bit integer values (normalization)
    port_out_gx.arr[:] = np.int32(magnitude_x)
    port_out_gx.set_valid()
    port_out_gy.arr[:] = np.int32(magnitude_y)
    port_out_gy.set_valid()


def compute_gradient_convolve(port_in: transferJobPorts.Port, port_out_name_gx: str, port_out_name_gy: str,
                              kernel_x: bytearray, kernel_y: bytearray) -> None:
    """
    Adds to specific output ports the result of convolution the image with 2 kernels(x and y)
    The function uses cv2.filter2D function
    :param port_in: image to apply convolution on
    :param port_out_name_gx: image resulted after convolution kernel for x and picture
    :param port_out_name_gy: image resulted after convolution kernel for y and picture
    :param kernel_x: kernel for x direction
    :param kernel_y: kernel for y direction
    :return: None
    """
    port_out_gx = get_port_from_wave(name=port_out_name_gx)
    port_out_gy = get_port_from_wave(name=port_out_name_gy)

    img = np.float64(port_in.arr)

    port_out_gx.arr[:] = convolve(input=img, weights=kernel_x)
    port_out_gy.arr[:] = convolve(input=img, weights=kernel_y)

    # Convert to signed 16 bit integer values (normalization)
    port_out_gx.set_valid()
    port_out_gy.set_valid()


def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def main_func_convolution(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in name: str, port_out_name_gx name: str, port_out_name_gy name: str, sobel_x name: str, sobel_y name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    INPUT_PORT_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    OUTPUT_GX_POS = 2
    # noinspection PyPep8Naming
    OUTPUT_GY_POS = 3
    # noinspection PyPep8Naming
    KERNEL_X_POS = 4
    # noinspection PyPep8Naming
    KERNEL_Y_POS = 5

    if len(param_list) != 6:
        log_error_to_console("KERNEL CONVOLUTION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in = get_port_from_wave(name=param_list[INPUT_PORT_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])

        if 'x' in param_list[KERNEL_X_POS] or 'y' in param_list[KERNEL_Y_POS]:
            kernel_x = eval('Application.Jobs.kernels.' + param_list[KERNEL_X_POS])
            kernel_y = eval('Application.Jobs.kernels.' + param_list[KERNEL_Y_POS])
        else:
            kernel_x = np.array(eval(param_list[KERNEL_X_POS]))
            kernel_y = np.array(eval(param_list[KERNEL_Y_POS]))

        if port_in.is_valid() is True:
            try:
                compute_gradient_filter_2d(port_in=port_in, port_out_name_gx=param_list[OUTPUT_GX_POS],
                                           port_out_name_gy=param_list[OUTPUT_GY_POS], kernel_x=kernel_x, kernel_y=kernel_y)
            except BaseException as error:
                log_error_to_console("KERNEL CONVOLUTION JOB NOK: ", str(error))
                pass
        else:
            return False

    return True


def main_func_deriche_convolution(param_list: list = None) -> bool:
    """
    Main function for deriche gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in name: str, port_out_name_gx name: str, port_out_name_gy name: str, sobel_x name: str, sobel_y name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    INPUT_PORT_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    KERNEL_ALPHA = 2
    # noinspection PyPep8Naming
    KERNEL_OMEGA = 3
    # noinspection PyPep8Naming
    OUTPUT_GX_POS = 4
    # noinspection PyPep8Naming
    OUTPUT_GY_POS = 5

    if len(param_list) != 6:
        log_error_to_console("KERNEL CONVOLUTION DERICHE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in = get_port_from_wave(name=param_list[INPUT_PORT_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])

        if port_in.is_valid() is True:
            try:
                port_out_gx = get_port_from_wave(name=param_list[OUTPUT_GX_POS])
                port_out_gy = get_port_from_wave(name=param_list[OUTPUT_GY_POS])

                magnitude_x = GradientDericheX(op=port_in.arr.copy(), alpha=param_list[KERNEL_ALPHA], omega=param_list[KERNEL_OMEGA])
                magnitude_y = GradientDericheY(op=port_in.arr.copy(), alpha=param_list[KERNEL_ALPHA], omega=param_list[KERNEL_OMEGA])

                # Convert to signed 16 bit integer values (normalization)
                port_out_gx.arr[:] = np.int16(magnitude_x)
                port_out_gx.set_valid()
                port_out_gy.arr[:] = np.int16(magnitude_y)
                port_out_gy.set_valid()
            except BaseException as error:
                log_error_to_console("KERNEL CONVOLUTION DERICHE JOB NOK: ", str(error))
                pass
        else:
            return False

    return True


def main_func_alternative(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in name: str, port_out_name_gx name: str, port_out_name_gy name: str,
                        sobel_x name: str, sobel_y name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    INPUT_PORT_POS = 0
    # noinspection PyPep8Naming
    OUTPUT_GX_POS = 1
    # noinspection PyPep8Naming
    OUTPUT_GY_POS = 2
    # noinspection PyPep8Naming
    KERNEL_X_POS = 3
    # noinspection PyPep8Naming
    KERNEL_Y_POS = 4

    if len(param_list) != 5:
        log_error_to_console("KERNEL CONVOLUTION  JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in = get_port_from_wave(param_list[INPUT_PORT_POS])

        if port_in.is_valid() is True:
            compute_gradient_convolve(port_in=port_in,
                                      port_out_name_gx=param_list[OUTPUT_GX_POS], port_out_name_gy=param_list[OUTPUT_GY_POS],
                                      kernel_x=eval('Application.Jobs.kernels.' + param_list[KERNEL_X_POS]),
                                      kernel_y=eval('Application.Jobs.kernels.' + param_list[KERNEL_Y_POS]))
        else:
            return False

        return True


def main_func_cross(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in name: str,
                        port_out_name_gn: str, port_out_name_gnw: str, port_out_name_gw: str, port_out_name_gsw: str,
                        port_out_name_gs: str, port_out_name_gse: str, port_out_name_ge: str, port_out_name_gne: str,
                        kernel name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    INPUT_PORT_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    OUTPUT_G1_POS = 2
    # noinspection PyPep8Naming
    OUTPUT_G2_POS = 3
    # noinspection PyPep8Naming
    OUTPUT_G3_POS = 4
    # noinspection PyPep8Naming
    OUTPUT_G4_POS = 5
    # noinspection PyPep8Naming
    OUTPUT_G5_POS = 6
    # noinspection PyPep8Naming
    OUTPUT_G6_POS = 7
    # noinspection PyPep8Naming
    OUTPUT_G7_POS = 8
    # noinspection PyPep8Naming
    OUTPUT_G8_POS = 9
    # noinspection PyPep8Naming
    KERNEL_POS = 10

    if len(param_list) != 11:
        log_error_to_console("KERNEL CROSS CONVOLUTION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in = get_port_from_wave(param_list[INPUT_PORT_POS], param_list[PORT_IN_WAVE_IMG])

        if 'x' in param_list[KERNEL_POS] or 'y' in param_list[KERNEL_POS]:
            kernel = eval('Application.Jobs.kernels.' + param_list[KERNEL_POS])
        else:
            kernel = np.array(eval(param_list[KERNEL_POS]))

        if port_in.is_valid() is True:
            try:
                compute_gradients_8_directions(port_in=port_in,
                                               port_out_name_gn=param_list[OUTPUT_G1_POS], port_out_name_gnw=param_list[OUTPUT_G2_POS],
                                               port_out_name_gw=param_list[OUTPUT_G3_POS], port_out_name_gsw=param_list[OUTPUT_G4_POS],
                                               port_out_name_gs=param_list[OUTPUT_G5_POS], port_out_name_gse=param_list[OUTPUT_G6_POS],
                                               port_out_name_ge=param_list[OUTPUT_G7_POS], port_out_name_gne=param_list[OUTPUT_G8_POS],
                                               kernel=kernel)
            except BaseException as error:
                log_error_to_console("KERNEL CROSS CONVOLUTION JOB NOK: ", str(error))
                pass

        else:
            return False

    return True


def main_func_frei_chen(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in name: str, wave_offset,
                        port_out_name_g1: str, port_out_name_g2: str, port_out_name_g3: str,
                        port_out_name_g4: str, port_out_name_g5: str, port_out_name_g6: str,
                        port_out_name_g7: str, port_out_name_g8: str, port_out_name_g9: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    INPUT_PORT_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_IN_KENEL_DILATION = 2
    # noinspection PyPep8Naming
    OUTPUT_G1_POS = 3
    # noinspection PyPep8Naming
    OUTPUT_G2_POS = 4
    # noinspection PyPep8Naming
    OUTPUT_G3_POS = 5
    # noinspection PyPep8Naming
    OUTPUT_G4_POS = 6
    # noinspection PyPep8Naming
    OUTPUT_G5_POS = 7
    # noinspection PyPep8Naming
    OUTPUT_G6_POS = 8
    # noinspection PyPep8Naming
    OUTPUT_G7_POS = 9
    # noinspection PyPep8Naming
    OUTPUT_G8_POS = 10
    # noinspection PyPep8Naming
    OUTPUT_G9_POS = 11

    if len(param_list) != 12:
        log_error_to_console("KERNEL CONVOLUTION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in = get_port_from_wave(param_list[INPUT_PORT_POS], param_list[PORT_IN_WAVE_IMG])

        if port_in.is_valid() is True:
            try:
                compute_gradients_frei_chen(port_in=port_in, dilation_factor=param_list[PORT_IN_KENEL_DILATION],
                                            port_out_name_g1=param_list[OUTPUT_G1_POS], port_out_name_g2=param_list[OUTPUT_G2_POS],
                                            port_out_name_g3=param_list[OUTPUT_G3_POS], port_out_name_g4=param_list[OUTPUT_G4_POS],
                                            port_out_name_g5=param_list[OUTPUT_G5_POS], port_out_name_g6=param_list[OUTPUT_G6_POS],
                                            port_out_name_g7=param_list[OUTPUT_G7_POS], port_out_name_g8=param_list[OUTPUT_G8_POS],
                                            port_out_name_g9=param_list[OUTPUT_G9_POS])
            except BaseException as error:
                log_error_to_console("KERNEL CONVOLUTION JOB NOK: ", str(error))
                pass
        else:
            return False

    return True


def main_func_navatia_babu(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in name: str, wave_offset,
                        port_out_name_g1: str, port_out_name_g2: str, port_out_name_g3: str,
                        port_out_name_g4: str, port_out_name_g5: str, port_out_name_g6: str,
                        port_out_name_g7: str, port_out_name_g8: str, port_out_name_g9: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    INPUT_PORT_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    OUTPUT_G1_POS = 2
    # noinspection PyPep8Naming
    OUTPUT_G2_POS = 3
    # noinspection PyPep8Naming
    OUTPUT_G3_POS = 4
    # noinspection PyPep8Naming
    OUTPUT_G4_POS = 5
    # noinspection PyPep8Naming
    OUTPUT_G5_POS = 6
    # noinspection PyPep8Naming
    OUTPUT_G6_POS = 7

    if len(param_list) != 8:
        log_error_to_console("KERNEL CONVOLUTION NAVATIA-BABU JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in = get_port_from_wave(param_list[INPUT_PORT_POS], param_list[PORT_IN_WAVE_IMG])

        if port_in.is_valid() is True:
            try:
                compute_gradients_navatia_babu(port_in=port_in,
                                               port_out_name_g1=param_list[OUTPUT_G1_POS], port_out_name_g2=param_list[OUTPUT_G2_POS],
                                               port_out_name_g3=param_list[OUTPUT_G3_POS], port_out_name_g4=param_list[OUTPUT_G4_POS],
                                               port_out_name_g5=param_list[OUTPUT_G5_POS], port_out_name_g6=param_list[OUTPUT_G6_POS])
            except BaseException as error:
                log_error_to_console("KERNEL CONVOLUTION NAVATIA-BABU JOB NOK: ", str(error))
                pass
        else:
            return False

    return True


if __name__ == "__main__":
    pass
