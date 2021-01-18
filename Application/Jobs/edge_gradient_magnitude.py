# noinspection PyPackageRequirements
import cv2
import numpy as np

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console

"""
Module handles first order magnitude derivatives orthogonal edge detection image jobs for the APPL block.
"""


def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def main_func(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in_gx name: str, wave_offset_gx, port_in_gy name: str, wave_offset_gy,
                        port_out_img name: str, port_out_ang: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_KERNEL_X_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_KERNEL_X = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_Y_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE_KERNEL_Y = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 4

    if len(param_list) != 5:
        log_error_to_console("EDGE DETECTION MAGNITUDE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in_gx = get_port_from_wave(name=param_list[PORT_KERNEL_X_POS], wave_offset=param_list[PORT_IN_WAVE_KERNEL_X])
        port_in_gy = get_port_from_wave(name=param_list[PORT_KERNEL_Y_POS], wave_offset=param_list[PORT_IN_WAVE_KERNEL_Y])
        port_out_img = get_port_from_wave(name=param_list[PORT_OUT_IMG])

        if (port_in_gx.is_valid() and port_in_gy.is_valid()) is True:
            try:
                result = np.hypot(port_in_gx.arr, port_in_gy.arr)
                port_out_img.arr[:] = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            except BaseException as error:
                log_error_to_console("EDGE DETECTION MAGNITUDE JOB NOK: ", str(error))
                pass
            port_out_img.set_valid()
        else:
            return False

        return True


def main_func_orientation(param_list: list = None) -> bool:
    """
    Main function for gradient angle calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in_gx name: str, wave_offset_gx, port_in_gy name: str, wave_offset_gy,
                        port_out_img name: str, port_out_ang: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_KERNEL_X_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_KERNEL_X = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_Y_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE_KERNEL_Y = 3
    # noinspection PyPep8Naming
    PORT_OUT_DIRECTION = 4

    if len(param_list) != 5:
        log_error_to_console("EDGE DETECTION ORIENTATION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in_gx = get_port_from_wave(name=param_list[PORT_KERNEL_X_POS], wave_offset=param_list[PORT_IN_WAVE_KERNEL_X])
        port_in_gy = get_port_from_wave(name=param_list[PORT_KERNEL_Y_POS], wave_offset=param_list[PORT_IN_WAVE_KERNEL_Y])

        port_out_img_gradient = get_port_from_wave(name=param_list[PORT_OUT_DIRECTION])

        if (port_in_gx.is_valid() and port_in_gy.is_valid()) is True:
            try:
                port_out_img_gradient.arr[:] = cv2.phase(x=port_in_gx.arr.astype(dtype=np.float),
                                                         y=port_in_gy.arr.astype(dtype=np.float),
                                                         angleInDegrees=True)
            except BaseException as error:
                log_error_to_console("EDGE DETECTION ORIENTATION JOB NOK: ", str(error))
                pass
            port_out_img_gradient.set_valid()
        else:
            return False

        return True


if __name__ == "__main__":
    pass
