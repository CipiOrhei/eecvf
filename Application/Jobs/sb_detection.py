# import what you need
import os.path

from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console
import config_main as CONFIG

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file

import json
import numpy as np
import cv2

"""
Module handles DESCRIPTION OF THE MODULE jobs for the APPL block.
"""


############################################################################################################################################
# Internal functions
############################################################################################################################################

def draw_line(image, line):
    start_point = (line[0][0], line[0][1])
    end_point = (line[1][0], line[1][1])

    color = (0, 255, 0)

    image2 = cv2.line(image, start_point, end_point, color, 2)
    return image2


def mean_y_line(line):
    return (line[0][0] + line[1][0]) // 2


def slope(line):
    try:
        return (np.abs(line[1][0] - line[0][0])) / (line[1][1] - line[0][1])
    except Exception:
        return 100


def isHorizontal(slope):
    return -5 <= slope <= 5

def gap(line1, line2):
    return line2[0][0] - line1[0][1]

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
    PORT_IN_MIN_GAP_HORIZONTAL_BOXES = 6
    # noinspection PyPep8Naming
    PORT_IN_MAX_GAP_HORIZONTAL_BOXES = 7
    # noinspection PyPep8Naming
    PORT_IN_MIN_GAP_VERTICAL_BOXES = 8
    # noinspection PyPep8Naming
    PORT_IN_MAX_GAP_VERTICAL_BOXES = 9
    # noinspection PyPep8Naming
    PORT_IN_MIN_LINE = 10
    # noinspection PyPep8Naming
    PORT_IS_DEBUG = 11
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 12
    # noinspection PyPep8Naming
    PORT_OUT_POS = 13
    # noinspection PyPep8Naming
    PORT_OUT_1_DEBUG = 14
    # noinspection PyPep8Naming
    PORT_OUT_2_DEBUG = 15
    # noinspection PyPep8Naming
    PORT_OUT_3_DEBUG = 16
    # verify that the number of parameters are OK.
    if (param_list[PORT_IS_DEBUG] is False and len(param_list) != 14) or (param_list[PORT_IS_DEBUG] is True and len(param_list) != 17):
        log_error_to_console("SB FROM LINES JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE])

        p_out_img = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_POS])

        if param_list[PORT_IS_DEBUG]:
            p_out_img_debug_1 = get_port_from_wave(name=param_list[PORT_OUT_1_DEBUG])
            p_out_img_debug_2 = get_port_from_wave(name=param_list[PORT_OUT_2_DEBUG])
            p_out_img_debug_3 = get_port_from_wave(name=param_list[PORT_OUT_3_DEBUG])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            # try:
            if True:
                lines = []

                # transform from lines with multiple points in line with 2 points
                for tmp_line in range(len(port_in.arr)):
                    start_point = [port_in.arr[tmp_line][0][0], port_in.arr[tmp_line][0][1]]
                    end_point = [0, 0]
                    idx = 0
                    if port_in.arr[tmp_line][idx][0] == 0 and port_in.arr[tmp_line][idx][1] == 0:
                        break

                    while True:
                        if port_in.arr[tmp_line][idx][0] == 0 and port_in.arr[tmp_line][idx][1] == 0:
                            break
                        end_point = [port_in.arr[tmp_line][idx][0], port_in.arr[tmp_line][idx][1]]
                        idx += 1
                    # lines.append([start_point, end_point])
                    lines.append([end_point, start_point])

                if param_list[PORT_IS_DEBUG]:
                    for el in lines:
                        start = (el[0][1], el[0][0])
                        end = (el[1][1], el[1][0])
                        cv2.line(p_out_img_debug_1.arr, start, end, (255, 0, 0), 1)

                # merge lines
                i = 0
                n = len(lines)
                max_horizontal_gap = param_list[PORT_IN_MAX_GAP_HORIZONTAL]
                min_horizontal_gap = param_list[PORT_IN_MIN_GAP_HORIZONTAL]
                max_vertical_gap = param_list[PORT_IN_MAX_GAP_VERTICAL]
                while i < n:
                    current_line = lines[i]
                    current_line_slope = slope(current_line)
                    j = i + 1
                    while j < n:
                        current_tmp_line = lines[j]
                        current_tmp_line_slope = slope(current_tmp_line)
                        horizontal_gap = current_tmp_line[0][1] - current_line[0][1]

                        if isHorizontal(current_line_slope) and isHorizontal(current_tmp_line_slope) and \
                                np.abs(mean_y_line(current_line) - mean_y_line(current_tmp_line)) <= max_vertical_gap and \
                                ((max_horizontal_gap >= horizontal_gap > min_horizontal_gap) or min_horizontal_gap > horizontal_gap > -max_horizontal_gap):
                            lines[i][1][1] = current_tmp_line[1][1]
                            lines[i][1][0] = mean_y_line(current_tmp_line)
                            lines.pop(j)
                            j -= 1
                            n -= 1
                        j += 1
                    i += 1

                min_line_length = param_list[PORT_IN_MIN_LINE]
                n = len(lines)
                i = 0
                while i < n:
                    line_len = np.abs(int(lines[i][0][1]) - int(lines[i][1][1]))
                    # print(line_len)
                    if line_len < min_line_length:
                        lines.pop(i)
                        i = 0
                        n -= 1
                    i += 1

                if param_list[PORT_IS_DEBUG]:
                    for el in lines:
                        start = (el[1][1], el[1][0])
                        end = (el[0][1], el[0][0])
                        cv2.line(p_out_img_debug_2.arr, start, end, (0, 0, 255), 1)

                ## end filter lines
                # show lines obtain at this step with red
                # for el in lines:
                #     start = (el[0][1], el[0][0])
                #     end = (el[1][1], el[1][0])
                #     cv2.line(p_out_img.arr, start, end, (255, 0, 0), 2)
                #
                #
                boxes = list()
                line_used = np.zeros(len(lines))
                for i in range(len(lines)):
                    min_overlap = lines[i][len(lines[i]) - 1][1] - lines[i][0][1] // 2
                    # min_overlap = 150
                    if isHorizontal(slope(lines[i])):
                        for j in range(len(lines)):
                            vertical_gap = (lines[j][0][0] + lines[j][len(lines[j]) - 1][0]) // 2 - (lines[i][0][0] + lines[j][len(lines[i]) - 1][0]) // 2  # gap between lines. compute mean value of first and last y of line
                            overlap = lines[i][0][1] - lines[j][len(lines[j]) - 1][1]

                            if overlap >= min_overlap and line_used[j] == 0 and param_list[PORT_IN_MIN_GAP_VERTICAL_BOXES] <= vertical_gap <= param_list[PORT_IN_MAX_GAP_VERTICAL_BOXES]:
                                line_used[i] = 1
                                line_used[j] = 1

                                top = mean_y_line(lines[i])
                                bottom = mean_y_line(lines[j])
                                left = lines[i][0][1] if lines[i][0][1] < lines[j][0][1] else lines[j][0][1]
                                right = lines[i][1][1] if lines[i][1][1] > lines[j][1][1] else lines[j][1][1]

                                boxes.append([[top, left], [bottom, right]])
                                break

                if param_list[PORT_IS_DEBUG]:
                    p_out_img_debug_3.arr = p_out_img_debug_2.arr.copy()

                    for box in boxes:
                        start = (box[0][1], box[0][0])
                        end = (box[1][1], box[1][0])
                        cv2.rectangle(p_out_img_debug_3.arr, start, end, (255, 0, 255), 2)

                p_out_img.set_valid()

                json_dict = dict()
                json_dict["asset"] = dict([
                    ("format", CONFIG.APPl_SAVE_PICT_EXTENSION),
                    ("name", CONFIG.APPL_INPUT_IMG_DIR[global_var_handler.FRAME]),
                    # {"path": os.path.join(CONFIG.APPL_INPUT_IMG_DIR, CONFIG.APPL_INPUT_IMG_DIR[CONFIG.APPL_NR_WAVES - 1])},
                    ("path", ""),
                    ("size", dict([("width", float(p_out_img.arr.shape[0])), ("height", float(p_out_img.arr.shape[1]))]))])

                region_list = list()
                for box in boxes:
                    height_rectangle = float(abs(box[1][0] - box[0][0]))
                    length_rectangle = float(abs(box[1][1] - box[0][1]))
                    d = dict([
                        ("id", ""),
                        ("type","RECTANGLE"),
                        ("tags", ["speedbump"]),
                        ("boundingBox", dict([("height",height_rectangle), ("width", length_rectangle),
                                             ("left", float(box[0][1])), ("top",float(box[0][0]))])),
                        ("points", [
                            dict([("x", float(box[0][1])), ("y", float(box[0][0]))]),
                            dict([("x", float(box[0][1] + length_rectangle)), ("y", float(box[0][0]))]),
                            dict([("x", float(box[1][1])), ("y", float(box[1][0]))]),
                            dict([("x", float(box[0][1])), ("y", float(box[0][0]) + height_rectangle)])
                                 ])])

                    region_list.append(d)
                json_dict["regions"] = region_list
                # print(json_dict)

                location = os.path.join(os.getcwd(), CONFIG.APPL_SAVE_LOCATION, p_out_img_debug_3.name)

                if not os.path.exists(os.path.join(os.getcwd(), location)):
                    os.makedirs(location)

                file_name = os.path.join(location, CONFIG.APPL_INPUT_IMG_DIR[global_var_handler.FRAME].split('.')[0] + '.json')
                file = open(file_name, 'w')
                print(json_dict)
                data_to_write = json.dumps(json_dict, indent=2)
                file.write(data_to_write)
                file.close()


                if param_list[PORT_IS_DEBUG]:
                    p_out_img_debug_1.set_valid()
                    p_out_img_debug_2.set_valid()
                    p_out_img_debug_3.set_valid()

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
                                   min_gap_horizontal_boxes: int = 10, max_gap_horizontal_boxes: int = 50, min_gap_vertical_boxes: int = 1, max_gap_vertical_boxes: int = 5,
                                   min_line_legth: int = 100,
                                   nr_sb: int = 1, debug: bool = True,
                                   port_img_output: str = None, port_detection_output: str = None,
                                   port_debug_1_output: str = None, port_debug_2_output: str = None, port_debug_3_output: str = None,
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
        details = 'LHG_{min_g_h}_{max_g_h}_LVG_{min_g_v}_{max_g_v}_BHG_{min_g_h_b}_{max_g_h_b}_BVG_{min_g_v_b}_{max_g_v_b}_MIN_LINE_{ml}'.format(
            min_g_h=min_gap_horizontal_lines.__str__().replace('.', '_'),
            max_g_h=max_gap_horizontal_lines.__str__().replace('.', '_'),
            min_g_v=min_gap_vertical_lines.__str__().replace('.', '_'),
            max_g_v=max_gap_vertical_lines.__str__().replace('.', '_'),
            min_g_h_b=min_gap_horizontal_boxes.__str__().replace('.', '_'),
            max_g_h_b=max_gap_horizontal_boxes.__str__().replace('.', '_'),
            min_g_v_b=min_gap_vertical_boxes.__str__().replace('.', '_'),
            max_g_v_b=max_gap_vertical_boxes.__str__().replace('.', '_'),
            ml=min_line_legth.__str__().replace('.', '_')
        )

        port_img_output = '{name}_{details}_{Input}'.format(name='SB_LINES_IMG', details=details, Input=port_input_name)
        port_detection_output = '{name}_{details}_{Input}'.format(name='SB_LINES', details=details, Input=port_input_name)

        if debug is True:
            port_debug_1_output = '{name}_{details}_{Input}'.format(name='SB_LINES_SIMPLY', details=details, Input=port_input_name)
            port_debug_2_output = '{name}_{details}_{Input}'.format(name='SB_LINES_MERGE', details=details, Input=port_input_name)
            port_debug_3_output = '{name}_{details}_{Input}'.format(name='SB_BOXES', details=details, Input=port_input_name)

    # size can be custom as needed
    port_img_output_name = transform_port_name_lvl(name=port_detection_output, lvl=level)
    port_img_output_name_size = transform_port_size_lvl(lvl=level, rgb=True)

    port_output_name = transform_port_name_lvl(name=port_img_output, lvl=level)

    if debug:
        port_debug_1_output_name = transform_port_name_lvl(name=port_debug_1_output, lvl=level)
        port_debug_1_output_name_size = transform_port_size_lvl(lvl=level, rgb=True)

        port_debug_2_output_name = transform_port_name_lvl(name=port_debug_2_output, lvl=level)
        port_debug_2_output_name_size = transform_port_size_lvl(lvl=level, rgb=True)

        port_debug_3_output_name = transform_port_name_lvl(name=port_debug_3_output, lvl=level)
        port_debug_3_output_name_size = transform_port_size_lvl(lvl=level, rgb=True)

    input_port_list = [input_port_name]

    main_func_list = [input_port_name, wave_offset,
                      min_gap_horizontal_lines, max_gap_horizontal_lines, min_gap_vertical_lines, max_gap_vertical_lines,
                      min_gap_horizontal_boxes, max_gap_horizontal_boxes, min_gap_vertical_boxes, max_gap_vertical_boxes,
                      min_line_legth,
                      debug,
                      port_img_output_name, port_output_name]

    if debug:
        main_func_list.append(port_debug_1_output_name)
        main_func_list.append(port_debug_2_output_name)
        main_func_list.append(port_debug_3_output_name)

    output_port_list = [(port_img_output_name, port_img_output_name_size, 'B', True),
                        (port_output_name, "(" + str(nr_sb) + ", 2, 2)", 'H', False)]
    if debug:
        output_port_list.append((port_debug_1_output_name, port_debug_1_output_name_size, 'B', True))
        output_port_list.append((port_debug_2_output_name, port_debug_2_output_name_size, 'B', True))
        output_port_list.append((port_debug_3_output_name, port_debug_3_output_name_size, 'B', True))

    job_name = job_name_create(action='SB FROM LINES', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=[port_img_output],
                                  main_func_name='main_func_sb_from_lines',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    if debug:
        return port_img_output, port_detection_output, port_debug_1_output, port_debug_2_output, port_debug_3_output
    else:
        return port_img_output, port_detection_output


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
