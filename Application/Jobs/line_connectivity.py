import numpy as np
# noinspection PyPackageRequirements
import cv2

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.port import Port
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_file, log_error_to_console


def process_edge_map(edge_map: Port.arr, port_name_output: Port.arr, connectivity: int):
    """
    # http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MORSE/connectivity.pdf
    :param edge_map : bitmap of edges
    :param port_name_output: port to hold output image
    :param connectivity: 8 or 4 for 8-way or 4-way connectivity respectively
    :return number of labels, average number of pixels per label, number of edges
    """
    p_out = get_port_from_wave(name=port_name_output)
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

    nr_edge_pixels = np.count_nonzero(labels)

    return num_labels, nr_edge_pixels / num_labels, nr_edge_pixels


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

    if len(param_list) != 4:
        log_error_to_console("EDGE LABEL JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1 = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])

        if p_in_1.is_valid() is True:
            try:
                nr_edge, average_px_edge, nr_edge_px = process_edge_map(edge_map=p_in_1.arr, port_name_output=param_list[PORT_OUT_POS],
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
