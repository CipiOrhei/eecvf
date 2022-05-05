# noinspection PyPackageRequirements
from typing import Tuple

# import thinning
# Do not delete used indirectly
# noinspection PyUnresolvedReferences
from Application.Frame import transferJobPorts
from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console
from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL, FILTERS
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file
import Application.Jobs.kernels

from Application.Jobs.external.EDLine.EdgeDrawing import EdgeDrawing
from Application.Jobs.ed_lines_modified import EdgeDrawing_modified
from Application.Jobs.external.EDLine.LineDetector import EDLine

import numpy as np
import cv2

############################################################################################################################################
# Init functions
############################################################################################################################################


def init_func_edge_drawing() -> JobInitStateReturn:
    """
    Init function for the draw edge algorithm
    :param port_list: Param needed list of port names [input, wave_of_input,  do_smoothing, edges_output, edge_map_output]
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def init_func_ed_lines() -> JobInitStateReturn:
    """
    Init function for the draw edge algorithm
    :return: INIT or NOT_INIT state for the job
    """

    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################


def main_edge_drawing_func(port_list: list = None) -> bool:
    """
    Edge segment detection algorithm that runs real-time and produces high quality edge segments, each of which is a linear pixel chain.
    Unlike traditional edge detectors, which work on the thresholded gradient magnitude cluster to determine edge elements, our method first
    spots sparse points along rows and columns called anchors, and then joins these anchors via a smart, heuristic edge tracing procedure,
    hence the name Edge Drawing (ED). ED produces edge maps that always consist of clean, perfectly contiguous, well-localized, one-pixel
    wide edges.
    :param port_list: Param needed list of port names [input, wave_of_input,  do_smoothing, edges_output, edge_map_output]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_SMOOTHING = 2
    # noinspection PyPep8Naming
    PORT_GAUSS_KERNEL_SIZE = 3
    # noinspection PyPep8Naming
    PORT_GAUSS_SIGMA = 4
    # noinspection PyPep8Naming
    PORT_GRAD_THR = 5
    # noinspection PyPep8Naming
    PORT_ANCHOR_THR = 6
    # noinspection PyPep8Naming
    PORT_SCAN_INTERVAL = 7
    # noinspection PyPep8Naming
    PORT_OUT_EDGES_POS = 8
    # noinspection PyPep8Naming
    PORT_OUT_EDGE_MAP_POS = 9

    # check if param OK
    if len(port_list) != 10:
        log_error_to_console("EDGE DRAWING JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE])
        p_out_edges = get_port_from_wave(name=port_list[PORT_OUT_EDGES_POS])
        p_out_edge_map = get_port_from_wave(name=port_list[PORT_OUT_EDGE_MAP_POS])

        if p_in.is_valid() is True:
            tmp_edge = np.array((1, 1))

            try:
                # parameters for Edge Drawing
                EDParam = {
                    # gaussian Smooth filter size if smoothed = False
                    'ksize': port_list[PORT_GAUSS_KERNEL_SIZE],
                    # gaussian smooth sigma ify smoothed = False
                    'sigma': port_list[PORT_GAUSS_SIGMA],
                    # threshold on gradient image
                    'gradientThreshold': port_list[PORT_GRAD_THR],
                    # threshold to determine the anchor
                    'anchorThreshold': port_list[PORT_ANCHOR_THR],
                    # scan interval, the smaller, the more detail
                    'scanIntervals': port_list[PORT_SCAN_INTERVAL]}

                ED_edge_map = EdgeDrawing(EDParam)

                edges, edges_map = ED_edge_map.EdgeDrawing(image=p_in.arr.copy(), smoothed=not (port_list[PORT_IN_SMOOTHING]))

                p_out_edge_map.arr[:] = edges_map
                p_out_edge_map.set_valid()

                for edge_id in range(len(edges)):
                    tmp_edge = np.array(edges[edge_id])
                    p_out_edges.arr[edge_id][:len(tmp_edge)] = tmp_edge

                p_out_edges.set_valid()
            except IndexError as error:
                log_error_to_console("ED_LINE JOB NOK! PLEASE ADJUST NUMBER OF EDGES: {e_a} < {e_m}".format(
                    e_a=len(edges), e_m=p_out_edges.arr.shape[0]))
                pass
            except ValueError as error:
                log_error_to_console("ED_LINE JOB NOK! PLEASE ADJUST NUMBER OF PIXELS OF EDGES: {e_a} < {e_m}".format(
                    e_a=tmp_edge.shape[0], e_m=p_out_edges.arr.shape[1]))
                pass
            except BaseException as error:
                log_error_to_console("ED_LINE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_edge_drawing_mod_func(port_list: list = None) -> bool:
    """
    Edge segment detection algorithm that runs real-time and produces high quality edge segments, each of which is a linear pixel chain.
    Unlike traditional edge detectors, which work on the thresholded gradient magnitude cluster to determine edge elements, our method first
    spots sparse points along rows and columns called anchors, and then joins these anchors via a smart, heuristic edge tracing procedure,
    hence the name Edge Drawing (ED). ED produces edge maps that always consist of clean, perfectly contiguous, well-localized, one-pixel
    wide edges.
    :param port_list: Param needed list of port names [input, wave_of_input,  do_smoothing, edges_output, edge_map_output]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    KERNEL_X_POS = 2
    # noinspection PyPep8Naming
    KERNEL_Y_POS = 3
    # noinspection PyPep8Naming
    PORT_GRAD_THR = 4
    # noinspection PyPep8Naming
    PORT_ANCHOR_THR = 5
    # noinspection PyPep8Naming
    PORT_SCAN_INTERVAL = 6
    # noinspection PyPep8Naming
    PORT_OUT_EDGES_POS = 7
    # noinspection PyPep8Naming
    PORT_OUT_EDGE_MAP_POS = 8

    # check if param OK
    if len(port_list) != 9:
        log_error_to_console("EDGE DRAWING JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE])
        p_out_edges = get_port_from_wave(name=port_list[PORT_OUT_EDGES_POS])
        p_out_edge_map = get_port_from_wave(name=port_list[PORT_OUT_EDGE_MAP_POS])

        if p_in.is_valid() is True:
            tmp_edge = np.array((1, 1))

            # if True:
            try:
                param_grad = 0
                if isinstance(port_list[PORT_GRAD_THR], str):
                    param_grad = get_port_from_wave(name=port_list[PORT_GRAD_THR], wave_offset=port_list[PORT_IN_WAVE]).arr[0]
                else:
                    param_grad = port_list[PORT_GRAD_THR]

                anchor_param = 0
                if isinstance(port_list[PORT_ANCHOR_THR], str):
                    anchor_param = get_port_from_wave(name=port_list[PORT_ANCHOR_THR], wave_offset=port_list[PORT_IN_WAVE]).arr[0]
                else:
                    anchor_param = port_list[PORT_ANCHOR_THR]

                if 'x' in port_list[KERNEL_X_POS] or 'y' in port_list[KERNEL_Y_POS]:
                    kernel_x = eval('Application.Jobs.kernels.' + port_list[KERNEL_X_POS])
                    kernel_y = eval('Application.Jobs.kernels.' + port_list[KERNEL_Y_POS])
                else:
                    kernel_x = np.array(eval(port_list[KERNEL_X_POS]))
                    kernel_y = np.array(eval(port_list[KERNEL_Y_POS]))

                # parameters for Edge Drawing
                EDParam = {
                    # threshold on gradient image
                    'gradientThreshold': param_grad,
                    # threshold to determine the anchor
                    'anchorThreshold': anchor_param,
                    # scan interval, the smaller, the more detail
                    'scanIntervals': port_list[PORT_SCAN_INTERVAL],
                    'kernel_x': kernel_x,
                    'kernel_y': kernel_y,
                }

                ED_edge_map = EdgeDrawing_modified(EDParam)

                edges, edges_map = ED_edge_map.EdgeDrawing(image=p_in.arr.copy())

                p_out_edge_map.arr[:] = edges_map
                p_out_edge_map.set_valid()

                for edge_id in range(len(edges)):
                    tmp_edge = np.array(edges[edge_id])
                    p_out_edges.arr[edge_id][:len(tmp_edge)] = tmp_edge

                p_out_edges.set_valid()
            except IndexError as error:
                log_error_to_console("ED_LINE JOB NOK! PLEASE ADJUST NUMBER OF EDGES: {e_a} < {e_m}".format(
                    e_a=len(edges), e_m=p_out_edges.arr.shape[0]))
                pass
            except ValueError as error:
                log_error_to_console("ED_LINE JOB NOK! PLEASE ADJUST NUMBER OF PIXELS OF EDGES: {e_a} < {e_m}".format(
                    e_a=tmp_edge.shape[0], e_m=p_out_edges.arr.shape[1]))
                pass
            except BaseException as error:
                log_error_to_console("ED_LINE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_ed_line_func(port_list: list = None) -> bool:
    """

    :param port_list: Param needed list of port names [input1, kernel_size, sigma, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_POS = 1
    # noinspection PyPep8Naming
    PORT_IN_SMOOTHING_POS = 2
    # noinspection PyPep8Naming
    PORT_GAUSS_KERNEL_SIZE_POS = 3
    # noinspection PyPep8Naming
    PORT_GAUSS_SIGMA_POS = 4
    # noinspection PyPep8Naming
    PORT_GRAD_THR_POS = 5
    # noinspection PyPep8Naming
    PORT_ANCHOR_THR_POS = 6
    # noinspection PyPep8Naming
    PORT_SCAN_INTERVAL_POS = 7
    # noinspection PyPep8Naming
    PORT_MIN_LINE_LEN_POS = 8
    # noinspection PyPep8Naming
    PORT_IN_FIT_ERR_THR_POS = 9
    # noinspection PyPep8Naming
    PORT_OUT_EDGES_POS = 10
    # noinspection PyPep8Naming
    PORT_OUT_EDGE_MAP_POS = 11
    # noinspection PyPep8Naming
    PORT_OUT_LINE_POS = 12
    # noinspection PyPep8Naming
    PORT_OUT_LINE_MAP_POS = 13

    # check if param OK
    if len(port_list) != 14:
        log_error_to_console("ED_LINE JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_POS])

        p_out_edges = get_port_from_wave(name=port_list[PORT_OUT_EDGES_POS])
        p_out_edge_map = get_port_from_wave(name=port_list[PORT_OUT_EDGE_MAP_POS])

        p_out_lines = get_port_from_wave(name=port_list[PORT_OUT_LINE_POS])
        p_out_map_lines = get_port_from_wave(name=port_list[PORT_OUT_LINE_MAP_POS])

        if p_in.is_valid() is True:
            try:
                # parameters for Edge Drawing
                EDParam = {
                    # gaussian Smooth filter size if smoothed = False
                    'ksize': port_list[PORT_GAUSS_KERNEL_SIZE_POS],
                    # gaussian smooth sigma ify smoothed = False
                    'sigma': port_list[PORT_GAUSS_SIGMA_POS],
                    # threshold on gradient image
                    'gradientThreshold': port_list[PORT_GRAD_THR_POS],
                    # threshold to determine the anchor
                    'anchorThreshold': port_list[PORT_ANCHOR_THR_POS],
                    # scan interval, the smaller, the more detail
                    'scanIntervals': port_list[PORT_SCAN_INTERVAL_POS]}

                ED_edge_map = EdgeDrawing(EDParam)

                edges, edges_map = ED_edge_map.EdgeDrawing(image=p_in.arr.copy(), smoothed=not (port_list[PORT_IN_SMOOTHING_POS]))

                lines = EDLine(edges=edges, minLineLen=port_list[PORT_MIN_LINE_LEN_POS],
                               lineFitErrThreshold=port_list[PORT_IN_FIT_ERR_THR_POS])

                tmp_line = np.array((1, 1))
                tmp_edge = np.array((1, 1))

                p_out_edge_map.arr[:] = edges_map
                p_out_edge_map.set_valid()

                for edge_id in range(len(edges)):
                    tmp_edge = np.array(edges[edge_id])
                    p_out_edges.arr[edge_id][:len(tmp_edge)] = tmp_edge

                p_out_edges.set_valid()

                # tmp_img = np.zeros((p_in.arr.shape[0], p_in.arr.shape[1], 3), dtype=np.uint8)

                for line_id in range(len(lines)):
                    tmp_line = np.array(lines[line_id])
                    p_out_lines.arr[line_id][:len(tmp_line)] = tmp_line

                    for el in tmp_line:
                        p_out_map_lines.arr[el[0], el[1]] = 255

                    # to activate if needed RGB lines.
                    # label_hue = np.uint8((line_id + 1) % 179)
                    # blank_ch = np.uint8(255 * label_hue)
                    # for el in tmp_line:
                    #     tmp_img[el[0], el[1], :] = (label_hue, blank_ch, blank_ch)

                # Converting cvt to BGR
                # p_out_map_lines.arr[:] = cv2.cvtColor(tmp_img, cv2.COLOR_HSV2BGR)

                p_out_lines.set_valid()
                p_out_map_lines.set_valid()

            except IndexError as error:
                log_error_to_console("ED_LINE JOB NOK! PLEASE ADJUST NUMBER OF EDGES: {e_a} < {e_m} OR LINES: {l_a} < {l_m}".format(
                    e_a=len(edges), e_m=p_out_edges.arr.shape[0], l_a=len(lines), l_m=p_out_lines.arr.shape[0]))
                pass
            except ValueError as error:
                log_error_to_console(
                    "ED_LINE JOB NOK! PLEASE ADJUST NUMBER OF PIXELS OF EDGES: {e_a} < {e_m} OR LINES: {l_a} < {l_m}".format(
                        e_a=tmp_edge.shape[0], e_m=p_out_edges.arr.shape[1], l_a=tmp_line.shape[0], l_m=p_out_lines.arr.shape[1]))
                pass
            except BaseException as error:
                log_error_to_console("ED_LINE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_ed_line_mod_func(port_list: list = None) -> bool:
    """

    :param port_list: Param needed list of port names [input1, kernel_size, sigma, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_POS = 1
    # noinspection PyPep8Naming
    KERNEL_X_POS = 2
    # noinspection PyPep8Naming
    KERNEL_Y_POS = 3
    # noinspection PyPep8Naming
    PORT_GRAD_THR_POS = 4
    # noinspection PyPep8Naming
    PORT_ANCHOR_THR_POS = 5
    # noinspection PyPep8Naming
    PORT_SCAN_INTERVAL_POS = 6
    # noinspection PyPep8Naming
    PORT_MIN_LINE_LEN_POS = 7
    # noinspection PyPep8Naming
    PORT_IN_FIT_ERR_THR_POS = 8
    # noinspection PyPep8Naming
    PORT_OUT_EDGES_POS = 9
    # noinspection PyPep8Naming
    PORT_OUT_EDGE_MAP_POS = 10
    # noinspection PyPep8Naming
    PORT_OUT_LINE_POS = 11
    # noinspection PyPep8Naming
    PORT_OUT_LINE_MAP_POS = 12

    # check if param OK
    if len(port_list) != 13:
        log_error_to_console("ED_LINE_MOD JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_POS])

        p_out_edges = get_port_from_wave(name=port_list[PORT_OUT_EDGES_POS])
        p_out_edge_map = get_port_from_wave(name=port_list[PORT_OUT_EDGE_MAP_POS])

        p_out_lines = get_port_from_wave(name=port_list[PORT_OUT_LINE_POS])
        p_out_map_lines = get_port_from_wave(name=port_list[PORT_OUT_LINE_MAP_POS])

        if p_in.is_valid() is True:
            try:
                param_grad = 0
                if isinstance(port_list[PORT_GRAD_THR_POS], str):
                    param_grad = get_port_from_wave(name=port_list[PORT_GRAD_THR_POS], wave_offset=port_list[PORT_IN_WAVE_POS]).arr[0]
                else:
                    param_grad = port_list[PORT_GRAD_THR_POS]

                anchor_param = 0
                if isinstance(port_list[PORT_ANCHOR_THR_POS], str):
                    anchor_param = get_port_from_wave(name=port_list[PORT_ANCHOR_THR_POS], wave_offset=port_list[PORT_IN_WAVE_POS]).arr[0]
                else:
                    anchor_param = port_list[PORT_ANCHOR_THR_POS]

                if 'x' in port_list[KERNEL_X_POS] or 'y' in port_list[KERNEL_Y_POS]:
                    kernel_x = eval('Application.Jobs.kernels.' + port_list[KERNEL_X_POS])
                    kernel_y = eval('Application.Jobs.kernels.' + port_list[KERNEL_Y_POS])
                else:
                    kernel_x = np.array(eval(port_list[KERNEL_X_POS]))
                    kernel_y = np.array(eval(port_list[KERNEL_Y_POS]))

                # parameters for Edge Drawing
                EDParam = \
                    {
                        # threshold on gradient image
                        'gradientThreshold': param_grad,
                        # threshold to determine the anchor
                        'anchorThreshold': anchor_param,
                        # scan interval, the smaller, the more detail
                        'scanIntervals': port_list[PORT_SCAN_INTERVAL_POS],
                        'kernel_x': kernel_x,
                        'kernel_y': kernel_y,
                }

                ED_edge_map = EdgeDrawing_modified(EDParam)

                edges, edges_map = ED_edge_map.EdgeDrawing(image=p_in.arr.copy())

                lines = EDLine(edges=edges, minLineLen=port_list[PORT_MIN_LINE_LEN_POS],
                               lineFitErrThreshold=port_list[PORT_IN_FIT_ERR_THR_POS])

                tmp_line = np.array((1, 1))
                tmp_edge = np.array((1, 1))

                p_out_edge_map.arr[:] = edges_map
                p_out_edge_map.set_valid()

                for edge_id in range(len(edges)):
                    tmp_edge = np.array(edges[edge_id])
                    p_out_edges.arr[edge_id][:len(tmp_edge),:] = tmp_edge

                p_out_edges.set_valid()

                # tmp_img = np.zeros((p_in.arr.shape[0], p_in.arr.shape[1], 3), dtype=np.uint8)

                for line_id in range(len(lines)):
                    tmp_line = np.array(lines[line_id])

                    p_out_lines.arr[line_id][:len(tmp_line), :] = tmp_line

                    for el in tmp_line:
                        p_out_map_lines.arr[el[0], el[1]] = 255

                    # to activate if needed RGB lines.
                    # label_hue = np.uint8((line_id + 1) % 179)
                    # blank_ch = np.uint8(255 * label_hue)
                    # for el in tmp_line:
                    #     tmp_img[el[0], el[1], :] = (label_hue, blank_ch, blank_ch)

                # Converting cvt to BGR
                # p_out_map_lines.arr[:] = cv2.cvtColor(tmp_img, cv2.COLOR_HSV2BGR)

                p_out_lines.set_valid()
                p_out_map_lines.set_valid()

            except IndexError as error:
                log_error_to_console("ED_LINE JOB NOK! PLEASE ADJUST NUMBER OF EDGES: {e_a} < {e_m} OR LINES: {l_a} < {l_m}".format(
                    e_a=len(edges), e_m=p_out_edges.arr.shape[0], l_a=len(lines), l_m=p_out_lines.arr.shape[0]))
                pass
            except ValueError as error:
                log_error_to_console(
                    "ED_LINE JOB NOK! PLEASE ADJUST NUMBER OF PIXELS OF EDGES: {e_a} < {e_m} OR LINES: {l_a} < {l_m}".format(
                        e_a=tmp_edge.shape[0], e_m=p_out_edges.arr.shape[1], l_a=tmp_line.shape[0], l_m=p_out_lines.arr.shape[1]))

                p_out_map_lines.arr[:]=np.zeros(shape=p_out_map_lines.arr.shape, dtype=p_out_map_lines.arr.dtype)
                p_out_map_lines.set_valid()
                pass
            except BaseException as error:
                log_error_to_console("ED_LINE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


############################################################################################################################################
# Job create functions
############################################################################################################################################


def do_edge_drawing_job(port_input_name: str,
                        max_edges: int = 5000, max_points_edge: int = 500,
                        gradient_thr: int = 36, anchor_thr: int = 8, scan_interval: int = 1,
                        do_smoothing: bool = True, gaussian_kernel_size: int = 3, gaussian_sigma: float = 1,
                        port_edge_map_name_output: str = None, port_edges_name_output: str = None,
                        level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> Tuple[str, str]:
    """
    Edge segment detection algorithm that runs real-time and produces high quality edge segments, each of which is a linear pixel chain.
    Unlike traditional edge detectors, which work on the thresholded gradient magnitude cluster to determine edge elements, our method first
    spots sparse points along rows and columns called anchors, and then joins these anchors via a smart, heuristic edge tracing procedure,
    hence the name Edge Drawing (ED). ED produces edge maps that always consist of clean, perfectly contiguous, well-localized, one-pixel
    wide edges.
    :param port_input_name: name of input port
    :param max_edges: max number of edges to hold in port
    :param max_points_edge: max number of points per edge
    :param gradient_thr: threshold on gradient image
    :param anchor_thr: threshold to determine the anchor
    :param scan_interval: scan interval, the smaller, the more detail
    :param do_smoothing: if we want to smooth the image
    :param gaussian_kernel_size: gaussian Smooth filter size if smoothed = False
    :param gaussian_sigma: gaussian smooth sigma ify smoothed = False
    :param port_edge_map_name_output: name of output port for edge map
    :param port_edges_name_output: name of output port for list of edge points
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_edge_map_name_output is None:
        port_edge_map_name_output = 'EDGE_DRAWING_THR_' + str(gradient_thr) + '_ANC_THR_' + str(anchor_thr) + '_SCAN_' + str(scan_interval)
        if do_smoothing is True:
            port_edge_map_name_output += '_GAUSS_S_' + str(gaussian_sigma).replace(".", "_") + '_K_' + str(gaussian_kernel_size) + '_' + port_input_name
        else:
            port_edge_map_name_output += '_' + port_input_name

    if port_edges_name_output is None:
        port_edges_name_output = 'EDGE_DRAWING_SEGMENTS_' + str(gradient_thr) + '_ANC_THR_' + str(anchor_thr) + '_SCAN_' + str(
            scan_interval)
        if do_smoothing is True:
            port_edges_name_output += '_GAUSS_S_' + str(gaussian_sigma).replace(".", "_") + '_' + port_input_name
        else:
            port_edges_name_output += '_' + port_input_name

    output_port_edge_map_name = transform_port_name_lvl(name=port_edge_map_name_output, lvl=level)
    output_port_edge_map_size = transform_port_size_lvl(lvl=level, rgb=False)

    output_port_edges_name = transform_port_name_lvl(name=port_edges_name_output, lvl=level)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, do_smoothing, gaussian_kernel_size, gaussian_sigma, gradient_thr, anchor_thr,
                      scan_interval, output_port_edges_name, output_port_edge_map_name]
    output_port_list = [(output_port_edges_name, "(" + str(max_edges) + "," + str(max_points_edge) + ", 2)", 'H', False),
                        (output_port_edge_map_name, output_port_edge_map_size, 'B', True)]

    job_name = job_name_create(action='Edge Drawing', input_list=input_port_list, wave_offset=[wave_offset], level=level)
    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_edge_drawing', init_func_param=None,
                                  main_func_name='main_edge_drawing_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_edge_map_name_output, port_edges_name_output


def do_edge_drawing_mod_job(port_input_name: str, operator: str,
                            max_edges: int = 5000, max_points_edge: int = 500,
                            gradient_thr: int = 36, anchor_thr: int = 8, scan_interval: int = 1,
                            port_edge_map_name_output: str = None, port_edges_name_output: str = None,
                            level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> Tuple[str, str]:
    """
    Edge segment detection algorithm that runs real-time and produces high quality edge segments, each of which is a linear pixel chain.
    Unlike traditional edge detectors, which work on the thresholded gradient magnitude cluster to determine edge elements, our method first
    spots sparse points along rows and columns called anchors, and then joins these anchors via a smart, heuristic edge tracing procedure,
    hence the name Edge Drawing (ED). ED produces edge maps that always consist of clean, perfectly contiguous, well-localized, one-pixel
    wide edges.
    Smoothing should be done independently of this job.
    :param port_input_name: name of input port
    :param operator: what operator we wish to use
    :param gradient_thr: threshold on gradient image
    :param anchor_thr: threshold to determine the anchor
    :param scan_interval: scan interval, the smaller, the more detail
    :param max_edges: max number of edges to hold in port
    :param max_points_edge: max number of points per edge
    :param port_edge_map_name_output: name of output port for edge map
    :param port_edges_name_output: name of output port for list of edge points
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    operator_job = operator.replace('_', ' ')
    kernel_x = operator.lower() + '_x'
    kernel_y = operator.lower() + '_y'

    if port_edge_map_name_output is None:
        port_edge_map_name_output = 'EDGE_DRAWING_MOD_THR_' + str(gradient_thr) + '_ANC_THR_' + str(anchor_thr) + '_SCAN_' + str(scan_interval)
        port_edge_map_name_output += '_' + operator_job.replace(' ', '_') + '_' + port_input_name

    if port_edges_name_output is None:
        port_edges_name_output = 'EDGE_DRAWING_MOD_SEGMENTS_' + str(gradient_thr) + '_ANC_THR_' + str(anchor_thr) + '_SCAN_' + str(
            scan_interval)
        port_edges_name_output += '_' + operator_job.replace(' ', '_') + '_' + port_input_name

    if isinstance(gradient_thr, str):
        gradient_thr += '_' + level

    if isinstance(anchor_thr, str):
        anchor_thr += '_' + level

    output_port_edge_map_name = transform_port_name_lvl(name=port_edge_map_name_output, lvl=level)
    output_port_edge_map_size = transform_port_size_lvl(lvl=level, rgb=False)

    output_port_edges_name = transform_port_name_lvl(name=port_edges_name_output, lvl=level)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel_x, kernel_y, gradient_thr, anchor_thr,
                      scan_interval, output_port_edges_name, output_port_edge_map_name]
    output_port_list = [(output_port_edges_name, "(" + str(max_edges) + "," + str(max_points_edge) + ", 2)", 'H', False),
                        (output_port_edge_map_name, output_port_edge_map_size, 'B', True)]

    job_name = job_name_create(action='Edge Drawing Modified', input_list=input_port_list, wave_offset=[wave_offset], level=level)
    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_edge_drawing', init_func_param=None,
                                  main_func_name='main_edge_drawing_mod_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_edge_map_name_output, port_edges_name_output


def do_ed_lines_job(port_input_name: str, min_line_length: int,
                    max_edges: int = 5000, max_points_edge: int = 500, max_lines: int = 5000, max_points_line: int = 500,
                    gradient_thr: int = 36, anchor_thr: int = 8, scan_interval: int = 1, line_fit_err_thr: int = 1,
                    do_smoothing: bool = True, gaussian_kernel_size: int = 0, gaussian_sigma: float = 0,
                    port_edge_map_name_output: str = None, port_edges_name_output: str = None,
                    port_lines_name_output: str = None, port_lines_img_output: str = None,
                    level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> Tuple[str, str, str, str]:
    """
    EDLines is comprised of three steps: (1) Given a grayscale image, we first run our fast, novel edge detector, the Edge Drawing (ED)
    algorithm, which produces a set of clean, contiguous chains of pixels, which we call edge segments. Edge segments intuitively correspond
    to object boundaries. (2)Next, we extract line segments from the generated pixel chains by means of a straightness criterion, i.e.,
    by the Least Squares Line Fitting Method. (3) Finally, a line validation step due to the Helmholtz principle  is used to eliminate
    false line segment detections.
    :param port_input_name: name of input port
    :param min_line_length: min number of pixel per line
    :param max_edges: max number of edges to hold in port
    :param max_points_edge: max number of points per edge
    :param max_lines: max number of lines to hold in port
    :param max_points_line: max number of points per line
    :param line_fit_err_thr: line fitting error
    :param gradient_thr: threshold on gradient image
    :param anchor_thr: threshold to determine the anchor
    :param scan_interval: scan interval, the smaller, the more detail
    :param do_smoothing: if we want to smooth the image
    :param gaussian_kernel_size: gaussian Smooth filter size if smoothed = False
    :param gaussian_sigma: gaussian smooth sigma ify smoothed = False
    :param port_edge_map_name_output: name of output port
    :param port_edges_name_output: name of output port
    :param port_lines_name_output: name of output port
    :param port_lines_img_output: name of output port
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port names: port_edges_name_output, port_edge_map_name_output, port_lines_name_output, port_lines_img_output
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_edge_map_name_output is None:
        port_edge_map_name_output = 'EDGE_DRAWING_THR_' + str(gradient_thr) + '_ANC_THR_' + str(anchor_thr) + '_SCAN_' + str(scan_interval)
        if do_smoothing is True:
            port_edge_map_name_output += '_GAUSS_S_' + str(gaussian_sigma).replace(".", "_") + '_' + port_input_name
        else:
            port_edge_map_name_output += '_' + port_input_name

    if port_edges_name_output is None:
        port_edges_name_output = 'EDGE_DRAWING_SEGMENTS_' + str(gradient_thr) + '_ANC_THR_' + str(anchor_thr) + '_SCAN_' + str(
            scan_interval)
        if do_smoothing is True:
            port_edges_name_output += '_GAUSS_S_' + str(gaussian_sigma).replace(".", "_") + '_' + port_input_name
        else:
            port_edge_map_name_output += '_' + port_input_name

    if port_lines_name_output is None:
        port_lines_name_output = 'ED_LINES_MIN_LEN_' + str(min_line_length) + '_LINE_FIT_ERR_' + str(
            line_fit_err_thr) + '_' + port_edges_name_output

    if port_lines_img_output is None:
        port_lines_img_output = 'ED_LINES_IMG_MIN_LEN_' + str(min_line_length) + '_LINE_FIT_ERR_' + str(
            line_fit_err_thr) + '_' + port_edges_name_output

    output_port_edge_map_name = transform_port_name_lvl(name=port_edge_map_name_output, lvl=level)
    output_port_edge_map_size = transform_port_size_lvl(lvl=level, rgb=False)

    output_port_line_img_name = transform_port_name_lvl(name=port_lines_img_output, lvl=level)
    output_port_line_img_size = transform_port_size_lvl(lvl=level, rgb=False)

    output_port_edges_name = transform_port_name_lvl(name=port_edges_name_output, lvl=level)
    output_port_lines_name = transform_port_name_lvl(name=port_lines_name_output, lvl=level)

    input_port_list = [input_port_name]

    main_func_list = [input_port_name, wave_offset,
                      do_smoothing, gaussian_kernel_size, gaussian_sigma, gradient_thr, anchor_thr, scan_interval,
                      min_line_length, line_fit_err_thr,
                      output_port_edges_name, output_port_edge_map_name,
                      output_port_lines_name, output_port_line_img_name]

    output_port_list = [(output_port_edges_name, "(" + str(max_edges) + "," + str(max_points_edge) + ", 2)", 'H', False),
                        (output_port_edge_map_name, output_port_edge_map_size, 'B', True),
                        (output_port_lines_name, "(" + str(max_lines) + "," + str(max_points_line) + ", 2)", 'H', False),
                        (output_port_line_img_name, output_port_line_img_size, 'B', True)]

    job_name = job_name_create(action='Ed_Lines', input_list=input_port_list, wave_offset=[wave_offset], level=level)
    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_ed_lines', init_func_param=None,
                                  main_func_name='main_ed_line_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_edge_map_name_output, port_edges_name_output, port_lines_name_output, port_lines_img_output


def do_ed_lines_mod_job(port_input_name: str, min_line_length: int, operator: str,
                        gradient_thr: int = 36, anchor_thr: int = 8, scan_interval: int = 1, line_fit_err_thr: int = 1,
                        max_edges: int = 5000, max_points_edge: int = 500, max_lines: int = 5000, max_points_line: int = 500,
                        port_edge_map_name_output: str = None, port_edges_name_output: str = None,
                        port_lines_name_output: str = None, port_lines_img_output: str = None,
                        level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> Tuple[str, str, str, str]:
    """
    EDLines is comprised of three steps: (1) Given a grayscale image, we first run our fast, novel edge detector, the Edge Drawing (ED)
    algorithm, which produces a set of clean, contiguous chains of pixels, which we call edge segments. Edge segments intuitively correspond
    to object boundaries. (2)Next, we extract line segments from the generated pixel chains by means of a straightness criterion, i.e.,
    by the Least Squares Line Fitting Method. (3) Finally, a line validation step due to the Helmholtz principle  is used to eliminate
    false line segment detections.
    :param port_input_name: name of input port
    :param line_fit_err_thr: line fitting error
    :param operator: what operator we wish to use
    :param gradient_thr: threshold on gradient image
    :param anchor_thr: threshold to determine the anchor
    :param scan_interval: scan interval, the smaller, the more detail
    :param min_line_length: min number of pixel per line
    :param max_edges: max number of edges to hold in port
    :param max_points_edge: max number of points per edge
    :param max_lines: max number of lines to hold in port
    :param max_points_line: max number of points per line
    :param port_edge_map_name_output: name of output port
    :param port_edges_name_output: name of output port
    :param port_lines_name_output: name of output port
    :param port_lines_img_output: name of output port
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port names: port_edges_name_output, port_edge_map_name_output, port_lines_name_output, port_lines_img_output
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    operator_job = operator.replace('_', ' ')
    kernel_x = operator.lower() + '_x'
    kernel_y = operator.lower() + '_y'

    if port_edge_map_name_output is None:
        port_edge_map_name_output = 'EDGE_DRAWING_MOD_THR_' + str(gradient_thr) + '_ANC_THR_' + str(anchor_thr) + '_SCAN_'\
                                    + str(scan_interval)
        port_edge_map_name_output += '_' + operator_job.replace(' ', '_') + '_' + port_input_name

    if port_edges_name_output is None:
        port_edges_name_output = 'EDGE_DRAWING_MOD_SEGMENTS_' + str(gradient_thr) + '_ANC_THR_' + str(anchor_thr) + '_SCAN_'\
                                 + str(scan_interval)
        port_edges_name_output += '_' + operator_job.replace(' ', '_') + '_' + port_input_name

    if port_lines_name_output is None:
        port_lines_name_output = 'ED_LINES_MIN_LEN_' + str(min_line_length) + '_LINE_FIT_ERR_' + str(line_fit_err_thr) + \
                                 '_' + port_edges_name_output

    if port_lines_img_output is None:
        port_lines_img_output = 'ED_LINES_IMG_MIN_LEN_' + str(min_line_length) + '_LINE_FIT_ERR_' + str(line_fit_err_thr) + \
                                '_' + port_edges_name_output

    if isinstance(gradient_thr, str):
        gradient_thr += '_' + level

    if isinstance(anchor_thr, str):
        anchor_thr += '_' + level

    output_port_edge_map_name = transform_port_name_lvl(name=port_edge_map_name_output, lvl=level)
    output_port_edge_map_size = transform_port_size_lvl(lvl=level, rgb=False)

    output_port_line_img_name = transform_port_name_lvl(name=port_lines_img_output, lvl=level)
    output_port_line_img_size = transform_port_size_lvl(lvl=level, rgb=False)

    output_port_edges_name = transform_port_name_lvl(name=port_edges_name_output, lvl=level)
    output_port_lines_name = transform_port_name_lvl(name=port_lines_name_output, lvl=level)

    input_port_list = [input_port_name]

    main_func_list = [input_port_name, wave_offset, kernel_x, kernel_y,
                      gradient_thr, anchor_thr, scan_interval,
                      min_line_length, line_fit_err_thr,
                      output_port_edges_name, output_port_edge_map_name,
                      output_port_lines_name, output_port_line_img_name]

    output_port_list = [(output_port_edges_name, "(" + str(max_edges) + "," + str(max_points_edge) + ", 2)", 'H', False),
                        (output_port_edge_map_name, output_port_edge_map_size, 'B', True),
                        (output_port_lines_name, "(" + str(max_lines) + "," + str(max_points_line) + ", 2)", 'H', False),
                        (output_port_line_img_name, output_port_line_img_size, 'B', True)]

    job_name = job_name_create(action='Ed_Lines', input_list=input_port_list, wave_offset=[wave_offset], level=level)
    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_ed_lines', init_func_param=None,
                                  main_func_name='main_ed_line_mod_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_edge_map_name_output, port_edges_name_output, port_lines_name_output, port_lines_img_output


if __name__ == "__main__":
    pass
