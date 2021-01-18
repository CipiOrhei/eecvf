# noinspection PyPackageRequirements
import cv2
# noinspection PyUnresolvedReferences
from Application.Frame import transferJobPorts
from Application.Frame.global_variables import JobInitStateReturn

from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_file, log_error_to_console


def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    log_to_file('Otsu output')
    return JobInitStateReturn(True)


def init_func() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def main_func(param_list: list = None) -> bool:
    """
    Main function for Otsu transformation job.
    :param param_list: Param needed to respect the following list:
                       [port_in image: image to run Otsu on
                          wave_offset,
                        port_out_img: port name to save resulted image
                        port_out_img: port name to save resulted threshold]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 2
    # noinspection PyPep8Naming
    PORT_OUT_VAL_POS = 3

    if len(param_list) != 4:
        log_error_to_console("OTSU TRANSFORMATION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(param_list[PORT_IN_POS], param_list[PORT_IN_WAVE])
        port_out_img = get_port_from_wave(param_list[PORT_OUT_IMG_POS])
        port_out_value = get_port_from_wave(param_list[PORT_OUT_VAL_POS])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                port_out_value.arr[:], port_out_img.arr[:] = cv2.threshold(src=port_in.arr.copy(), thresh=0, maxval=255,
                                                                           type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                port_out_value.set_valid()
                port_out_img.set_valid()

                log_to_file(port_out_value.arr[0].__str__())
            except BaseException as error:
                log_to_file('')
                log_error_to_console("OTSU TRANSFORMATION JOB NOK: ", str(error))
                pass
        else:
            log_to_file('')
            return False

        return True


def main_func_thresholding(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in image: image to run Otsu on
                        port_out_img: port name to save resulted image
                        port_out_img: port name to save resulted threshold]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_THRESHOLD_VALUE_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_THRESHOLD_TYPE_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 4

    if len(param_list) != 5:
        log_error_to_console("IMAGE THRESHOLD JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(param_list[PORT_IN_POS], param_list[PORT_IN_WAVE])
        port_out = get_port_from_wave(param_list[PORT_OUT_IMG])

        port_in_value = param_list[PORT_IN_THRESHOLD_VALUE_POS]

        if type(port_in_value) == type(str()):
            port_in_value = get_port_from_wave(param_list[PORT_IN_THRESHOLD_VALUE_POS]).arr

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                ret, port_out.arr[:] = cv2.threshold(src=port_in.arr, type=eval(param_list[PORT_IN_THRESHOLD_TYPE_POS]), maxval=255, thresh=port_in_value)
                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("IMAGE THRESHOLD JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_func_adaptive_thresholding(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in image: image to run Otsu on
                        port_out_img: port name to save resulted image
                        port_out_img: port name to save resulted threshold]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_THRESHOLD_METHOD_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_THRESHOLD_TYPE_POS = 3
    # noinspection PyPep8Naming
    PORT_IN_BLOCK_SIZE_POS = 4
    # noinspection PyPep8Naming
    PORT_IN_CONSTANT_SIZE_POS = 5
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 6

    if len(param_list) != 7:
        log_error_to_console("IMAGE ADAPTIVE THRESHOLD JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(param_list[PORT_IN_POS], param_list[PORT_IN_WAVE])
        port_out = get_port_from_wave(param_list[PORT_OUT_IMG])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                port_out.arr[:] = cv2.adaptiveThreshold(src=port_in.arr, thresholdType=eval(param_list[PORT_IN_THRESHOLD_TYPE_POS]),
                                                          maxValue=255, adaptiveMethod=eval(param_list[PORT_IN_THRESHOLD_METHOD_POS]),
                                                          blockSize=param_list[PORT_IN_BLOCK_SIZE_POS],
                                                          C=param_list[PORT_IN_CONSTANT_SIZE_POS])
                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("IMAGE ADAPTIVE THRESHOLD JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


if __name__ == "__main__":
    pass
