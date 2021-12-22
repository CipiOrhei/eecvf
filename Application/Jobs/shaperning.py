# import what you need
import os

from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console
import config_main as CONFIG
from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file
from Utils.plotting import plot_histogram_grey_image
import cv2

"""
Module handles sharpening algorithm for an image jobs for the APPL block.
"""

############################################################################################################################################
# Internal functions
############################################################################################################################################

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
                p_out.arr[:] = cv2.equalizeHist(port_in.arr.copy())

                if param_list[PORT_SAVE_HIST] == True:
                    plot_histogram_grey_image(image=port_in.arr.copy(), name_folder=port_in.name, picture_name=global_var_handler.PICT_NAME.split('.')[0],
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

############################################################################################################################################
# Job create functions
############################################################################################################################################

def do_histogram_equalization_job(port_input_name: str, save_histogram = True,
                                  port_img_output: str = None,
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
    :param wave_offset: wave of input port, please correlate with each input port name parameter
    :return: Name of output port or ports
    """
    # Do this for each input port this function has
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_img_output is None:
        port_img_output = 'HIST_EQUAL_{Input}'.format(Input=port_input_name)

    # size can be custom as needed
    port_img_output_name = transform_port_name_lvl(name=port_img_output, lvl=level)
    port_img_output_name_size = transform_port_size_lvl(lvl=level, rgb=False)

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


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
