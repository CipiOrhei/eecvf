import os
# noinspection PyPackageRequirements
import cv2

import config_main

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave, reshape_ports
from Utils.log_handler import log_to_file, log_to_console, is_error, log_error_to_console
from Application.Frame.global_variables import global_var_handler

"""
Module handles retrieval image jobs for the APPL block.
"""


def get_image_cv(path: str, port_raw_image: str) -> None:
    """
    Get's the picture accordingly to frame and populate the ports with the raw data color.
    :param port_raw_image: Name of port input of raw image
    :param path: path to picture
    :return: None
    """
    port_image = get_port_from_wave(name=port_raw_image)
    # TODO use yield
    if True:
    # try:
        img = cv2.imread(filename=path)
        height, width = img.shape[:2]

        if width != global_var_handler.WIDTH_L0 or height != global_var_handler.HEIGHT_L0:
            log_to_console("RAW PICTURE SIZE NOT OK! PICTURE SIZE REDONE")
            global_var_handler.HEIGHT_L0 = height
            global_var_handler.WIDTH_L0 = width
            global_var_handler.recalculate_pyramid_level_values()
            # noinspection PyUnresolvedReferences
            reshape_ports(size_array=global_var_handler.SIZE_ARRAY)

        port_image.arr[:] = img
        port_image.set_valid()

    # except BaseException as error:
    #     is_error()
    #     log_error_to_console('RAW PICTURE NOK TO READ: ' + str(path), str(error))
    #     port_image.set_invalid()
    #     pass


def init_func() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    log_to_file('Frame')

    if config_main.APPL_INPUT == config_main.IMAGE_INPUT:
        log_to_file('Image Name')

    log_to_file('Raw Pict Size')

    # noinspection PyUnresolvedReferences
    return JobInitStateReturn(True if global_var_handler.NR_PICTURES != 0 else False)


# noinspection PyUnresolvedReferences
def main_func(param_list: list = None) -> bool:
    """
    Main function for retrieving pictures.
    The job will populate the respective ports with the raw image.
    :param param_list: raw picture port,
                       raw picture grey port
    :return: True if the job executed OK.
    """
    # index of param
    # noinspection PyPep8Naming
    PORT_RAW_PICT = 0

    # check if param OK
    if len(param_list) != 1:
        log_error_to_console("GET FRAME MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        get_image_cv(path=os.path.join(config_main.APPL_INPUT_DIR, config_main.APPL_INPUT_IMG_DIR[global_var_handler.FRAME]),
                     port_raw_image=param_list[PORT_RAW_PICT])

        global_var_handler.PICT_NAME = config_main.APPL_INPUT_IMG_DIR[global_var_handler.FRAME]

        log_to_file(str(global_var_handler.FRAME))
        log_to_file(global_var_handler.PICT_NAME)
        log_to_file(global_var_handler.STR_L0_SIZE)

        return True


def main_func_satellite(param_list: list = None) -> bool:
    """
    Main function for retrieving satellite images
    The job will populate the respective ports with the raw satellite image and greyscale image
    :param param_list: raw picture port,
                       raw picture grey port
    :return: True if the job executed OK.
    """
    # index of param
    PORT_RAW_PICT = 0

    # check if param OK
    if len(param_list) != 1:
        log_error_to_console("GET FRAME MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:

        path = os.path.join(config_main.APPL_INPUT_DIR, config_main.APPL_INPUT_IMG_DIR[global_var_handler.FRAME])
        port_image = get_port_from_wave(param_list[PORT_RAW_PICT])
        try:
            img = cv2.imread(path)
            height, width = img.shape[:2]

            if width != global_var_handler.WIDTH_L0 or height != global_var_handler.HEIGHT_L0:
                log_to_console("RAW PICTURE SIZE NOT OK! PICTURE SIZE REDONE")
                global_var_handler.HEIGHT_L0 = height
                global_var_handler.WIDTH_L0 = width
                global_var_handler.recalculate_pyramid_level_values()
                reshape_ports(global_var_handler.SIZE_ARRAY)

            port_image.arr[:] = img
            port_image.set_valid()

        except BaseException as error:
            is_error()
            log_error_to_console('RAW PICTURE NOK TO READ: ' + str(path), str(error))
            port_image.set_invalid()
            pass

        global_var_handler.PICT_NAME = config_main.APPL_INPUT_IMG_DIR[global_var_handler.FRAME]

        log_to_file(str(global_var_handler.FRAME))
        log_to_file(global_var_handler.PICT_NAME)
        log_to_file(global_var_handler.STR_L0_SIZE)

        return True


def main_func_video(param_list: list = None) -> bool:
    """
    Main function for retrieving frames from a video.
    The job will populate the respective ports with the raw image.
    :param param_list: raw picture port
    :return: True if the job executed OK.
    """
    # index of param
    # noinspection PyPep8Naming
    PORT_RAW_PICT = 0

    # check if param OK
    if len(param_list) != 1:
        log_error_to_console("GET FRAME VIDEO MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_image = get_port_from_wave(name=param_list[PORT_RAW_PICT])

        try:
            # noinspection PyUnresolvedReferences
            success, port_image.arr[:] = global_var_handler.VIDEO.read()
            if success is True:
                port_image.set_valid()
        except BaseException as error:
            is_error()
            # noinspection PyUnresolvedReferences
            log_error_to_console('RAW PICTURE NOK TO READ: ' + str(global_var_handler.VIDEO.__str__()), str(error))
            port_image.set_invalid()
            pass

        # noinspection PyUnresolvedReferences
        log_to_file(str(global_var_handler.FRAME))
        # noinspection PyUnresolvedReferences
        log_to_file(global_var_handler.STR_L0_SIZE)

        return True


def main_func_video_camera(param_list: list = None) -> bool:
    """
    Main function for retrieving frames from a video camera stream.
    The job will populate the respective ports with the raw image.
    :param param_list: raw picture port
    :return: True if the job executed OK.
    """
    # index of param
    # noinspection PyPep8Naming
    PORT_RAW_PICT = 0

    # check if param OK
    if len(param_list) != 1:
        log_error_to_console("GET FRAME VIDEO CAPTURE MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_image = get_port_from_wave(name=param_list[PORT_RAW_PICT])

        try:
            # noinspection PyUnresolvedReferences
            success, port_image.arr[:] = global_var_handler.VIDEO.read()
            if success is True:
                port_image.set_valid()
        except BaseException as error:
            is_error()
            # noinspection PyUnresolvedReferences
            log_error_to_console('RAW PICTURE NOK TO READ: ' + str(global_var_handler.VIDEO.__str__()), str(error))
            port_image.set_invalid()
            pass

        # noinspection PyUnresolvedReferences
        log_to_file(str(global_var_handler.FRAME))
        # noinspection PyUnresolvedReferences
        log_to_file(global_var_handler.STR_L0_SIZE)

        return True


if __name__ == "__main__":
    pass
