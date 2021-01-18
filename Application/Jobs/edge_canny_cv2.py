# noinspection PyPackageRequirements
import cv2

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame import transferJobPorts
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import is_error, log_error_to_console


class CANNY_CONFIG:
    """
    Class to hold canny configs
    """
    # fix threshold of low = 60, high = 90
    FIX_THRESHOLD = 0
    # threshold high = otsu and low = otsu * 0.5
    OTSU_HALF = 1
    # threshold calculation using sigma and median
    OTSU_MEDIAN_SIGMA = 2
    # threshold calculation using sigma and median
    MEDIAN_SIGMA = 3
    # threshold calculation ratio
    RATIO_THRESHOLD = 4
    # threshold calculation ratio
    RATIO_MEAN = 5
    # manual threshold
    MANUAL_THRESHOLD = 6


def calc_threshold(config: int = CANNY_CONFIG.FIX_THRESHOLD, val: int = 0) -> tuple:
    """
    :param config: choose configuration from CANNY_CONFIG
    :param val: some configuration need a value to calculate
    :return: lower and high threshold
    """
    if config == CANNY_CONFIG.FIX_THRESHOLD:
        return 80, 170

    elif config == CANNY_CONFIG.OTSU_HALF:
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.403.5666&rep=rep1&type=pdf#page=120
        return int(val * 0.5), val

    elif config == CANNY_CONFIG.OTSU_MEDIAN_SIGMA or CANNY_CONFIG.MEDIAN_SIGMA:
        # https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
        # https://ieeexplore.ieee.org/abstract/document/5476095
        sigma = 0.33
        lower = int(max(0, int(1.0 - sigma) * val))
        upper = int(min(255, int(1.0 + sigma) * val))
        return lower, upper

    elif config == CANNY_CONFIG.RATIO_MEAN:
        # http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
        lower = int(max(0, int(0.66 * val)))
        upper = int(min(255, int(1.33 * val)))

        return lower, upper

    elif config == CANNY_CONFIG.RATIO_THRESHOLD:
        # http://justin-liang.com/tutorials/canny/
        high = int(val * 0.7)
        low = int(high * 0.3)

        return low, high


def do_canny(port_in_gx: transferJobPorts.Port, port_in_gy: transferJobPorts.Port,
             port_out: transferJobPorts.Port, th_config: int = 0, thresh: int = 0,
             th_low: int = 0, th_high: int = 0) -> None:
    """
    Calculates canny transformation with kernels passed
    :param port_in_gx: port for matrix resulted for x kernel
    :param port_in_gy: port for matrix resulted for y kernel
    :param port_out: port for resulting image after canny processing
    :param th_config: configuration to use for canny thresholds
    :param th_low: manual low threshold
    :param th_high: manual high threshold
    :param thresh: otsu threshold

    :return: None
    """
    if th_config is not CANNY_CONFIG.MANUAL_THRESHOLD:
        th_low, th_high = calc_threshold(th_config, thresh)

    try:
        # noinspection PyArgumentList
        port_out.arr[:] = cv2.Canny(dx=port_in_gx.arr, dy=port_in_gy.arr, threshold1=th_low, threshold2=th_high, L2gradient=True)
        port_out.set_valid()
    except BaseException as error:
        is_error()
        log_error_to_console('CORRUPT JPEG DATA ' + str(port_out.get_name()), str(error))
        port_out.set_invalid()
        pass

    # cv2.imshow(str(port_out_name), port_out.arr)
    # cv2.waitKey(0)


def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def main_func_var_trh(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param neededs to respect the following list:
                       [port_in_gx name: str, port_in_gy name: str,
                        port_out_name name: str
                        port_config configuration to do canny with: str
                        port_in_value: otsu value or other thresholds: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_KERNEL_X_POS = 0
    # noinspection PyPep8Naming
    PORT_KERNEL_Y_POS = 1
    # noinspection PyPep8Naming
    PORT_WAVE_IN = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3
    # noinspection PyPep8Naming
    PORT_CONFIG_POS = 4
    # noinspection PyPep8Naming
    PORT_LOW_THR = 5
    # noinspection PyPep8Naming
    PORT_HIGH_THR = 6
    # noinspection PyPep8Naming
    PORT_VAL_POS = 7

    if len(param_list) != 8:
        log_error_to_console("CANNY EDGE DETECTION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in_gx = get_port_from_wave(param_list[PORT_KERNEL_X_POS], param_list[PORT_WAVE_IN])
        port_in_gy = get_port_from_wave(param_list[PORT_KERNEL_Y_POS], param_list[PORT_WAVE_IN])

        port_out = get_port_from_wave(param_list[PORT_OUT_POS])

        if param_list[PORT_VAL_POS] is not None:
            value = get_port_from_wave(param_list[PORT_VAL_POS]).arr
        else:
            value = 0

        if param_list[PORT_CONFIG_POS] is not None:
            config = eval(param_list[PORT_CONFIG_POS])
        else:
            config = CANNY_CONFIG.MANUAL_THRESHOLD

        if (port_in_gx.is_valid() and port_in_gy.is_valid()) is True:
            try:
                do_canny(port_in_gx, port_in_gy, port_out, config, value, param_list[PORT_LOW_THR], param_list[PORT_HIGH_THR])
            except BaseException as error:
                is_error()
                log_error_to_console('CANNY PARAM_LIST NOK: ' + str(param_list[PORT_VAL_POS]), str(error))
        else:
            return False

        return True


if __name__ == "__main__":
    pass
