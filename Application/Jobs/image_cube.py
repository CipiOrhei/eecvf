# import what you need
import os

import numpy as np
import cv2

import config_main
from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL, APPL_INPUT_IMG_DIR
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file

from Application.Frame.global_variables import global_var_handler

"""
Module handles DESCRIPTION OF THE MODULE jobs for the APPL block.
"""

############################################################################################################################################
# Internal functions
############################################################################################################################################

############################################################################################################################################
# Init functions
############################################################################################################################################

cube_matrix = list()

# define a init function, function that will be executed at the begging of the wave
def init_func_global(param_list) -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################

# define a main function, function that will be executed at the begging of the wave
def main_func(param_list: list = None) -> bool:
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
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_OUT_NAME = 2
    # noinspection PyPep8Naming
    PORT_OUT_LOCATION = 3
    # verify that the number of parameters are OK.
    if len(param_list) != 4:
        log_error_to_console("IMAGE CUBE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            if True:
            # try:
                global cube_matrix
                cube_matrix.append(port_in.arr)

                if len(cube_matrix) == len(APPL_INPUT_IMG_DIR):
                    # Cast the list into a numpy array
                    img_array = np.array(cube_matrix)
                    for line in range(img_array.shape[1]):
                        if len(img_array.shape) == 4:
                            line_image = img_array[:, line, :, :]
                        else:
                            line_image = img_array[:, line, :]
                        # img_location = 'Logs/process_cube/IMG_CUBE_GRAY_RAW_LINE_L0'
                        img_location = param_list[PORT_OUT_LOCATION] + '/' + 'LINE_SLICING_' + param_list[PORT_OUT_NAME]
                        if not os.path.exists(img_location):
                            os.makedirs(img_location)
                        cv2.imwrite(img_location + '/' + '{:08d}'.format(line) + config_main.APPl_SAVE_PICT_EXTENSION, line_image)

                    for col in range(img_array.shape[2]):
                        if len(img_array.shape) == 4:
                            col_image = img_array[:, :, col, :]
                            col_image = np.transpose(col_image, (1, 0, 2))
                        else:
                            col_image = img_array[:, :, col]
                            col_image = np.transpose(col_image)
                        # img_location = 'Logs/process_cube/IMG_CUBE_GRAY_RAW_COL_L0'
                        img_location = param_list[PORT_OUT_LOCATION] + '/' + 'COLUMN_SLICING_' + param_list[PORT_OUT_NAME]
                        if not os.path.exists(img_location):
                            os.makedirs(img_location)
                        cv2.imwrite(img_location + '/' + '{:08d}'.format(col) + config_main.APPl_SAVE_PICT_EXTENSION, col_image)

            # except BaseException as error:
            #     log_error_to_console("JOB_NAME JOB NOK: ", str(error))
            #     pass
        else:
            return False

        return True

############################################################################################################################################
# Job create functions
############################################################################################################################################

def create_image_cube(port_input_name: str, location_to_save: str,
                      port_img_output: str = None, is_rgb: bool = False,
                      level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    User-interface of {job}
    Add details of job
    Please add paper from feature
    Add as many other parameters needed
    :param port_input_name:  One or several input ports
    :param port_img_output:
    :param level: Level of input port, please correlate with each input port name parameter
    :param wave_offset: wave of input port, please correlate with each input port name parameter
    :return: Name of output port or ports
    """
    # Do this for each input port this function has
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_img_output is None:
        port_img_output = 'IMG_CUBE_{Input}'.format(Input=port_input_name)

    # size can be custom as needed
    port_img_output_name = transform_port_name_lvl(name=port_img_output, lvl=level)
    port_img_output_name_size = transform_port_size_lvl(lvl=level, rgb=True)

    input_port_list = [input_port_name]
    init_func_param = [is_rgb]
    main_func_list = [input_port_name, wave_offset, port_img_output_name, location_to_save]
    # output_port_list = [(port_img_output_name, port_img_output_name_size, 'B', True)]
    output_port_list = []

    job_name = job_name_create(action='IMG_CUBE_{Input}'.format(Input=port_input_name), input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=init_func_param,
                                  main_func_name='main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_img_output


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
