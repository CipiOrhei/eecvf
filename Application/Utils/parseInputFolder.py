import os
# noinspection PyPackageRequirements
import cv2

import config_main

from Utils.log_handler import log_to_console, log_error_to_console
from Application.Frame.global_variables import global_var_handler

"""
Module handles the input folder functionalities for the APPL block
"""


# directory where images are stored
MAX_IMAGE_HEIGHT = 0
MAX_IMAGE_WIDTH = 0


def get_camera_capture() -> None:
    """
    Updates the global variable for camera capture input mode
    :return: None
    """
    global_var_handler.VIDEO = cv2.VideoCapture(0)
    global_var_handler.NR_PICTURES = config_main.APPL_NR_FRAMES_CAPTURE
    global_var_handler.WIDTH_L0 = int(cv2.VideoCapture.get(global_var_handler.VIDEO, cv2.CAP_PROP_FRAME_WIDTH))
    global_var_handler.HEIGHT_L0 = int(cv2.VideoCapture.get(global_var_handler.VIDEO, cv2.CAP_PROP_FRAME_HEIGHT))
    global_var_handler.recalculate_pyramid_level_values()

    # noinspection PyUnresolvedReferences
    log_to_console('CONFIGURATION UPDATE TO: MAX PICTURE SIZE {} AND NR PICTURES {}'.
                   format(global_var_handler.STR_L0_SIZE, global_var_handler.NR_PICTURES))


def get_video_capture() -> None:
    """
    Updates the global variable for video input mode
    :return: None
    """
    global_var_handler.VIDEO = cv2.VideoCapture(config_main.APPL_INPUT_VIDEO)
    global_var_handler.NR_PICTURES = int(cv2.VideoCapture.get(global_var_handler.VIDEO, cv2.CAP_PROP_FRAME_COUNT))
    global_var_handler.WIDTH_L0 = int(cv2.VideoCapture.get(global_var_handler.VIDEO, cv2.CAP_PROP_FRAME_WIDTH))
    global_var_handler.HEIGHT_L0 = int(cv2.VideoCapture.get(global_var_handler.VIDEO, cv2.CAP_PROP_FRAME_HEIGHT))
    global_var_handler.recalculate_pyramid_level_values()

    # noinspection PyUnresolvedReferences
    log_to_console('CONFIGURATION UPDATE TO: MAX PICTURE SIZE {} AND NR PICTURES {}'.
                   format(global_var_handler.STR_L0_SIZE, global_var_handler.NR_PICTURES))


def release_video() -> None:
    """
    Releases video capture stream.
    :return: None
    """
    # noinspection PyUnresolvedReferences
    global_var_handler.VIDEO.release()


def get_images_from_dir(directory: str) -> None:
    """
    Updates the global variable images_in_directory with all the frames inside.
    :param directory: source directory for the input pictures
    :return: None
    """
    for dir_name, dir_names, file_names in os.walk(directory):
        # array that stores all names
        for filename in file_names:
            config_main.APPL_INPUT_IMG_DIR.append(filename)


def clear_input_img_dir():
    """
    Clear the list of images from dir
    :return: None
    """
    config_main.APPL_INPUT_IMG_DIR.clear()


def find_big_picture() -> None:
    """
    Searches the biggest picture from the input file
    Updates the global variables HEIGHT and WIDTH
    :return: None
    """
    global MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT
    for image in config_main.APPL_INPUT_IMG_DIR:
        try:
            img = cv2.imread(os.path.join(os.getcwd(), config_main.APPL_INPUT_DIR, image))
            height, width = img.shape[:2]

            if MAX_IMAGE_HEIGHT < height:
                MAX_IMAGE_HEIGHT = height
            if MAX_IMAGE_WIDTH < width:
                MAX_IMAGE_WIDTH = width
        except BaseException as error:
            log_error_to_console('IMAGE READ FIND BIGGEST PICTURE', str(error))
            pass


def get_picture_size_and_number() -> None:
    """
    Maps the picture from the input dir.
    Updates data in global_variables.py(HEIGHT_L0,WIDTH_L0,NR_PICTURES)
    :return: None
    """
    get_images_from_dir(os.path.join(os.getcwd(), config_main.APPL_INPUT_DIR))

    if len(config_main.APPL_INPUT_IMG_DIR) != 0:
        img = cv2.imread(os.path.join(os.getcwd(), config_main.APPL_INPUT_DIR, config_main.APPL_INPUT_IMG_DIR[0]))
        height, width = img.shape[:2]
        global_var_handler.WIDTH_L0 = width
        global_var_handler.HEIGHT_L0 = height
        global_var_handler.recalculate_pyramid_level_values()
        global_var_handler.NR_PICTURES = len(config_main.APPL_INPUT_IMG_DIR)

        # noinspection PyUnresolvedReferences
        log_to_console('CONFIGURATION UPDATE TO: MAX PICTURE SIZE {} AND NR PICTURES {}'.
                       format(global_var_handler.STR_L0_SIZE, global_var_handler.NR_PICTURES))
    else:
        log_error_to_console('IMAGE FOLDER IS EMPTY')


if __name__ == "__main__":
    pass
