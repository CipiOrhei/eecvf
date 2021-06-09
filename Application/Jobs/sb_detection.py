# import what you need

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file

import numpy as np
import cv2

"""
Module handles DESCRIPTION OF THE MODULE jobs for the APPL block.
"""

############################################################################################################################################
# Internal functions
############################################################################################################################################

############################################################################################################################################
# Init functions
############################################################################################################################################

# define a init function, function that will be executed at the begging of the wave
def init_func_global(param) -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :return: INIT or NOT_INIT state for the job
    """
    log_to_file(param[0])
    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################

# define a main function, function that will be executed at the begging of the wave
def main_func_sb_from_lines(param_list: list = None) -> bool:
    """
    Main function for {job} calculation job.
    :param param_list: Param needed to respect the following list:
                       [enumerate list]
    :return: True if the job executed OK.
    """
    # variables for position of param needed
    # ex:

    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_MIN_GAP_HORIZONTAL = 2
    # noinspection PyPep8Naming
    PORT_IN_MAX_GAP_HORIZONTAL = 3
    # noinspection PyPep8Naming
    PORT_IN_MIN_GAP_VERTICAL = 4
    # noinspection PyPep8Naming
    PORT_IN_MAX_GAP_VERTICAL = 5
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 6
    # noinspection PyPep8Naming
    PORT_OUT_POS = 7
    # verify that the number of parameters are OK.
    if len(param_list) != 8:
        log_error_to_console("SB FROM LINES JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE])

        p_out_img = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_POS])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            # try:
            if True:
                    line_idx = 0
                    lines = list()

                    for tmp_line in port_in.arr:
                        start_point = tmp_line[0]
                        end_point = [0, 0]
                        idx = 0
                        if tmp_line[idx][0] == 0 and tmp_line[idx][1] == 0:
                            break

                        while True:
                            if tmp_line[idx][0] == 0 and tmp_line[idx][1] == 0:
                                break
                            end_point = tmp_line[idx]
                            idx += 1

                        lines.append([start_point, end_point])

                    line_used = np.zeros(len(lines))
                    boxes = list()
                    for i in range(len(lines)):
                        min_overlap = lines[i][len(lines[i]) - 1][0] - lines[i][0][0] // 2
                        for j in range(len(lines) - 1, i, -1):
                            vertical_gap = (lines[j][0][1] + lines[j][len(lines[j]) - 1][1]) // 2 - (lines[i][0][1] + lines[j][len(lines[i]) - 1][1]) // 2  # gap between lines. compute mean value of first and last y of line
                            overlap = lines[i][0][0] - lines[j][len(lines[j]) - 1][0]

                            if overlap >= min_overlap and param_list[PORT_IN_MIN_GAP_HORIZONTAL] <= min_overlap <= param_list[PORT_IN_MAX_GAP_HORIZONTAL] and line_used[j] == 0\
                                and param_list[PORT_IN_MIN_GAP_VERTICAL] <= vertical_gap <= param_list[PORT_IN_MAX_GAP_VERTICAL]:
                                line_used[j] = 1

                                boxes.append([lines[i][0], lines[j][len(lines[j]) - 1]])

                                # new box between line i and j'

                    # p_out.arr = lines
                    log_to_file(p_out.arr.__str__())
                    p_out.set_valid()


                    for box in boxes:
                        start = (box[0][1], box[0][0])
                        end = (box[1][1], box[1][0])
                        cv2.rectangle(p_out_img.arr, start, end, (255,0,255), 2)

                    for el in lines:
                        start = (el[0][1], el[0][0])
                        end = (el[1][1], el[1][0])
                        cv2.line(p_out_img.arr, start, end, (0,0,255), 2)

                    p_out_img.set_valid()

            # except BaseException as error:
            #     log_error_to_console("JOB_NAME JOB NOK: ", str(error))
            #     pass
        else:
            return False

        return True

############################################################################################################################################
# Job create functions
############################################################################################################################################

def do_sb_detection_from_lines_job(port_input_name: str,
                                   min_gap_horizontal_lines: int = 10, max_gap_horizontal_lines: int = 50, min_gap_vertical_lines: int = 1, max_gap_vertical_lines: int = 5,
                                   nr_sb: int = 1,
                                   port_img_output: str = None, port_detection_output: str = None,
                                   level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    TBD
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
        port_img_output = '{name}_MIN_HG_{min_g_h}_MAX_HG_{max_g_h}_MIN_VG_{min_g_v}_MAX_HG_{max_g_v}_{Input}'.format(name='SB_LINES_IMG',
                                                                                                                      min_g_h=min_gap_horizontal_lines.__str__().replace('.', '_'),
                                                                                                                      max_g_h=max_gap_horizontal_lines.__str__().replace('.', '_'),
                                                                                                                      min_g_v=min_gap_vertical_lines.__str__().replace('.', '_'),
                                                                                                                      max_g_v=max_gap_vertical_lines.__str__().replace('.', '_'),
                                                                                                                      Input=port_input_name)

        port_detection_output = '{name}_MIN_HG_{min_g_h}_MAX_HG_{max_g_h}_MIN_VG_{min_g_v}_MAX_HG_{max_g_v}_{Input}'.format(name='SB_LINES',
                                                                                                                      min_g_h=min_gap_horizontal_lines.__str__().replace('.', '_'),
                                                                                                                      max_g_h=max_gap_horizontal_lines.__str__().replace('.', '_'),
                                                                                                                      min_g_v=min_gap_vertical_lines.__str__().replace('.', '_'),
                                                                                                                      max_g_v=max_gap_vertical_lines.__str__().replace('.', '_'),
                                                                                                                      Input=port_input_name)

    # size can be custom as needed
    port_img_output_name = transform_port_name_lvl(name=port_detection_output, lvl=level)
    port_img_output_name_size = transform_port_size_lvl(lvl=level, rgb=True)

    port_output_name = transform_port_name_lvl(name=port_img_output, lvl=level)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, min_gap_horizontal_lines, max_gap_horizontal_lines, min_gap_vertical_lines, max_gap_vertical_lines, port_img_output_name, port_output_name]
    output_port_list = [(port_img_output_name, port_img_output_name_size, 'B', True),
                        (port_output_name, "(" + str(nr_sb) + ", 2, 2)", 'H', False)]

    job_name = job_name_create(action='SB FROM LINES', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=[port_img_output],
                                  main_func_name='main_func_sb_from_lines',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_img_output, port_detection_output


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
