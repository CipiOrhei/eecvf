# noinspection PyPackageRequirements
import cv2
import numpy as np

from skimage.filters import threshold_multiotsu

# noinspection PyUnresolvedReferences
from Application.Frame import transferJobPorts
from Application.Frame.global_variables import JobInitStateReturn

from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_file, log_error_to_console
from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL, FILTERS
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file

"""
Module handles thresholding images
"""


############################################################################################################################################
# Init functions
############################################################################################################################################

def init_func_global(param_list) -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    log_to_file('Otsu output of ' + param_list[0])
    return JobInitStateReturn(True)


def init_func() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################


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
                ret, port_out.arr[:] = cv2.threshold(src=port_in.arr, type=eval(param_list[PORT_IN_THRESHOLD_TYPE_POS]), maxval=255,
                                                     thresh=port_in_value)
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


def main_func_multi_lvl_otsu(param_list: list = None) -> bool:
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
    PORT_IN_NR_CLASSES = 2
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 3
    # check last value + 1(difference pos vs index) + nr of values - 1 (n classes -> n-1 values)
    if len(param_list) != PORT_OUT_IMG + 1 + param_list[PORT_IN_NR_CLASSES] - 1:
        log_error_to_console("MULTI LEVEL OTSU JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(param_list[PORT_IN_POS], param_list[PORT_IN_WAVE])
        port_out = get_port_from_wave(param_list[PORT_OUT_IMG])

        value_list = []
        # classes - 1 = values
        for lvl in range(1, param_list[PORT_IN_NR_CLASSES]):
            value_list.append(get_port_from_wave(param_list[PORT_OUT_IMG + lvl]))

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                t = threshold_multiotsu(port_in.arr.copy(), param_list[PORT_IN_NR_CLASSES])

                for el in range(len(t)):
                    value_list[el].arr[:] = t[el]
                    value_list[el].set_valid()

                # Using the threshold values, we generate the three regions.
                tmp = np.digitize(port_in.arr.copy(), bins=t)
                port_out.arr[:] = cv2.normalize(tmp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                port_out.set_valid()
                log_to_file(t.__str__())
            except BaseException as error:
                log_to_file('')
                log_error_to_console("IMAGE ADAPTIVE THRESHOLD JOB NOK: ", str(error))
                pass
        else:
            log_to_file('')
            return False

        return True


############################################################################################################################################
# Job create functions
############################################################################################################################################


def do_multi_otsu_job(port_input_name: str, number_of_classes: int = 3,
                      port_output_name: str = None,
                      level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0):
    """
    Multi-Otsu calculates several thresholds, determined by the number of desired classes. The default number of classes is 3: for obtaining
    three classes, the algorithm returns two threshold values. They are represented by a red line in the histogram below.
    Works on grayscale only
    http://smile.ee.ncku.edu.tw/old/Links/MTable/ResearchPaper/papers/2001/A%20fast%20algorithm%20for%20multilevel%20%20thresholding.pdf
    :param port_input_name: name of port on which we desire to apply Otsu on
    :param number_of_classes: name of desired classes.
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'OTSU_MULTI_LEVEL_' + port_input_name

    output_port_img = transform_port_name_lvl(name=port_output_name + '_IMG', lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)
    # output_port_value = transform_port_name_lvl(name=port_output_name + '_VALUE', lvl=level)

    input_port_list = [input_port]
    main_func_list = [input_port, wave_offset, number_of_classes, output_port_img]

    output_port_list = [(output_port_img, output_port_size, 'B', True)]

    for th in range(1, number_of_classes):
        port_name = transform_port_name_lvl(name=port_output_name + '_VALUE_' + str(th), lvl=level)
        output_port_list.append((port_name, '1', 'B', False))
        # output_port_list.append((output_port_value + '_' + str(th), '1', 'B', False))
        main_func_list.append(port_name)

    job_name = job_name_create(action='Multi-Otsu Thresholding', input_list=input_port_list, wave_offset=[wave_offset, wave_offset],
                               level=level)
    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=[port_output_name],
                                  main_func_name='main_func_multi_lvl_otsu',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


if __name__ == "__main__":
    pass
