import os
# noinspection PyPackageRequirements
import cv2
import numpy as np

import config_main

from Application.Frame.global_variables import global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console

"""
Module handles the image handing of the APPL block
"""

ROTATE = False


def rotate_picture() -> None:
    """
    Rotate images when shown.
    :return: None
    """
    global ROTATE
    ROTATE = True


def show_pictures() -> None:
    """
    Shows ports passed in port_names for the time specified
    :return: None
    """
    if config_main.APPL_SHOW_PICT:
        for port_name in config_main.APPL_SHOW_LIST:
            port = get_port_from_wave(port_name)
            try:
                img = port.arr.copy()
                if ROTATE:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                if img.dtype != np.uint8:
                    log_error_to_console('SHOW PICTURE FORMAT ERROR!',
                                         'IMAGE CONVERTED FROM {} TO uint8. CHECK SAVED FILE FOR CORRECT IMAGE!'.format(img.dtype))
                cv2.imshow(port_name, img.astype(np.uint8))
            except BaseException as error:
                log_error_to_console('SHOW PICTURE ' + str(port_name), str(error))
                pass

        cv2.waitKey(config_main.APPL_SHOW_TIME)


def save_pict_to_file() -> None:
    """
    Saves ports to file
    :return: None
    """
    location = config_main.APPL_SAVE_LOCATION
    # noinspection PyUnresolvedReferences
    name = global_var_handler.PICT_NAME

    if config_main.APPL_SAVE_PICT:
        for img in config_main.APPL_SAVE_PICT_LIST:
            img_location = os.path.join(location, img)
            port = get_port_from_wave(img)
            if not os.path.exists(img_location):
                os.makedirs(img_location)
            try:
                if port.is_valid() is True:
                    extension = config_main.APPl_SAVE_PICT_EXTENSION
                    if name is None:
                        # noinspection PyUnresolvedReferences
                        if config_main.APPL_SAVE_JOB_NAME:
                            cv2.imwrite(img_location + '/' + str(global_var_handler.FRAME) + '_' + port.get_name() + extension, port.arr)
                        else:
                            cv2.imwrite(img_location + '/' + str(global_var_handler.FRAME) + extension, port.arr)
                    else:
                        name = name.split('.')[0] + extension
                        if config_main.APPL_SAVE_JOB_NAME:
                            cv2.imwrite(img_location + '/' + port.get_name() + '_' + name, port.arr)
                        else:
                            cv2.imwrite(img_location + '/' + name, port.arr)
            except BaseException as error:
                log_error_to_console('SAVE PICTURE ' + str(img), str(error))
                pass


if __name__ == "__main__":
    pass
