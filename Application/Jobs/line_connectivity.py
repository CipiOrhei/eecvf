import math

# noinspection PyPackageRequirements
from typing import Tuple
import numpy as np
# noinspection PyPackageRequirements
import cv2

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.port import Port
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_file, log_error_to_console

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file


############################################################################################################################################
# Internal functions
############################################################################################################################################

def process_edge_map(edge_map: Port.arr, port_name_output: Port.arr, port_name_labels_output: Port.arr, connectivity: int):
    """
    # http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MORSE/connectivity.pdf
    :param edge_map : bitmap of edges
    :param port_name_output: port to hold output image
    :param connectivity: 8 or 4 for 8-way or 4-way connectivity respectively
    :return number of labels, average number of pixels per label, number of edges
    """
    p_out = get_port_from_wave(name=port_name_output)
    p_out_labels = get_port_from_wave(name=port_name_labels_output)
    # threshold image to be sure that all edges have 255 value
    ret, edge_map = cv2.threshold(src=edge_map, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    edge_map = np.uint8(edge_map)

    num_labels, labels = cv2.connectedComponents(image=edge_map, connectivity=connectivity)

    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    p_out.arr[:] = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    p_out.arr[label_hue == 0] = 0

    p_out.set_valid()

    p_out_labels.arr[:] = labels
    p_out_labels.set_valid()

    nr_edge_pixels = np.count_nonzero(labels)

    return num_labels, nr_edge_pixels / num_labels, nr_edge_pixels

############################################################################################################################################
# Init functions
############################################################################################################################################

def init_edge_label(param_list: list = None) -> JobInitStateReturn:
    """
    Init function for the job
    :param param_list: list of PORT to be written in the csv file
    :return: INIT or NOT_INIT state for the job
    """
    log_to_file('Nr Edges ' + param_list[0])
    log_to_file('AVG px/edge ' + param_list[0])
    log_to_file('Nr Edge px ' + param_list[0])

    return JobInitStateReturn(True)


# define a init function, function that will be executed at the begging of the wave
def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)

############################################################################################################################################
# Main functions
############################################################################################################################################

def create_edge_label_map(param_list: list = None) -> bool:
    """
    Calculates the maximum pixel value
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, wave_in: int,
                        port_out: image RGB of edges]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_CONNECTIVITY_POS = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_LABELS_POS = 4

    if len(param_list) != 5:
        log_error_to_console("EDGE LABEL JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1 = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])

        if p_in_1.is_valid() is True:
            try:
                nr_edge, average_px_edge, nr_edge_px = process_edge_map(edge_map=p_in_1.arr, port_name_output=param_list[PORT_OUT_POS], port_name_labels_output=param_list[PORT_OUT_LABELS_POS],
                                                                        connectivity=param_list[PORT_CONNECTIVITY_POS])
                log_to_file(str(nr_edge))
                log_to_file(str(average_px_edge))
                log_to_file(str(nr_edge_px))
            except BaseException as error:
                log_error_to_console("EDGE LABEL JOB NOK: ", str(error))
                log_to_file('')
                log_to_file('')
                log_to_file('')
                pass
        else:
            log_to_file('')
            log_to_file('')
            log_to_file('')
            return False

        return True


# define a main function, function that will be executed at the begging of the wave
def main_func_line_filtering(param_list: list = None) -> bool:
    """
    Main function for {job} calculation job.
    :param param_list: Param needed to respect the following list:
                       [enumerate list]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_THETA = 2
    # noinspection PyPep8Naming
    PORT_IN_DEVIATION = 3
    # noinspection PyPep8Naming
    PORT_OUTPUT_LINE = 4
    # noinspection PyPep8Naming
    PORT_OUTPUT_LINE_IMG = 5

    # verify that the number of parameters are OK.
    if len(param_list) != 6:
        log_error_to_console("LINE FILTERING JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        p_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])

        p_out_lines = get_port_from_wave(name=param_list[PORT_OUTPUT_LINE])
        p_out_lines_img = get_port_from_wave(name=param_list[PORT_OUTPUT_LINE_IMG])

        # check if port's you want to use are valid
        if p_in.is_valid() is True:
            try:
                value = math.tan(math.radians(param_list[PORT_IN_THETA]))
                grade = param_list[PORT_IN_DEVIATION]
                min_value = value - math.radians(grade)
                max_value = value + math.radians(grade)

                line_idx = 0

                for line in p_in.arr:
                    start_point = line[0]
                    end_point = [0, 0]
                    idx = 0
                    if line[idx][0] == 0 and line[idx][1] == 0:
                        break

                    while True:
                        if line[idx][0] == 0 and line[idx][1] == 0:
                            break
                        end_point = line[idx]
                        idx += 1


                    line_slope = (np.abs(int(end_point[0]) - int(start_point[0])) / (int(end_point[1]) - int(start_point[1])))
                    # line_slope = (end_point[0] - start_point[0]) / (end_point[1] - start_point[1])

                    if min_value < line_slope < max_value:
                        p_out_lines.arr[line_idx][:] = line

                        for el in p_out_lines.arr[line_idx]:
                            p_out_lines_img.arr[el[0], el[1]] = 255

                        line_idx += 1

                p_out_lines.set_valid()
                p_out_lines_img.set_valid()
            except BaseException as error:
                log_error_to_console("LINE FILTERING JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


############################################################################################################################################
# Job create functions
############################################################################################################################################

def do_line_theta_filtering_job(port_input_name: str, theta_value: int, deviation_theta: float = 10,
                                nr_lines: int = 50, nr_pt_line: int = 50,
                                port_output: str = None, port_img_output: str = None,
                                level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> Tuple[str, str]:
    """
    Filters lines accordingly to a theta value. 0 for horizontal
    :param port_input_name:  One or several input ports
    :param theta_value:  theta value
    :param deviation_theta:  accepted deviation of theta value
    :param nr_lines:  number of lines to keep at the end
    :param nr_pt_line:  number of points per lines to keep at the end
    :param port_output: port of lines
    :param port_img_output: port of image of lines kept
    :param level: Level of input port, please correlate with each input port name parameter
    :param wave_offset: wave of input port, please correlate with each input port name parameter
    :return: Name of output port or ports
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_img_output is None:
        port_output = '{name}_{theta}_{theta_value}_{theta_procent}_{theta_p_value}_{Input}'.format(name='LINE_FILTERING',
                                                                                                    theta='T', theta_value=theta_value.__str__().replace('.', '_'),
                                                                                                    theta_procent='D', theta_p_value=deviation_theta,
                                                                                                    Input=port_input_name)
        port_img_output = '{name}_{theta}_{theta_value}_{theta_procent}_{theta_p_value}_{Input}'.format(name='LINE_FILTERING_IMG',
                                                                                                        theta='T', theta_value=theta_value.__str__().replace('.', '_'),
                                                                                                        theta_procent='D', theta_p_value=deviation_theta,
                                                                                                        Input=port_input_name)

    output_port_line_img_name = transform_port_name_lvl(name=port_img_output, lvl=level)
    output_port_line_img_size = transform_port_size_lvl(lvl=level, rgb=False)
    port_line_output_name = transform_port_name_lvl(name=port_output, lvl=level)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, theta_value, deviation_theta, port_line_output_name, output_port_line_img_name]
    output_port_list = [(port_line_output_name, "(" + str(nr_lines) + "," + str(nr_pt_line) + ", 2)", 'H', False),
                        (output_port_line_img_name, output_port_line_img_size, 'B', True)]

    job_name = job_name_create(action='LINE FILTERING', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_line_filtering',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output, port_img_output

if __name__ == "__main__":
    # If you want to run something stand-alone
    pass