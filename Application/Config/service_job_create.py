import glob
import os
import shutil

import config_main

from Application.Config.create_config import created_port_list
from Utils.log_handler import log_setup_info_to_console
from Application.Utils.image_handler import rotate_picture

"""
All the service jobs from this block have an wrapper in this module. This module handles the transition between the user interface and 
services of the EECVF.
"""


def set_output_image_folder(folder: str) -> None:
    """
    Service that sets up the location of image output folder
    :param folder: location to set
    :return: None
    """
    config_main.APPL_SAVE_LOCATION = folder

    log_setup_info_to_console('IMAGE FOLDER OUTPUT:{}'.format(os.path.join(os.getcwd(), config_main.APPL_INPUT_DIR)))


def set_input_image_folder(folder: str) -> None:
    """
    Service that sets up the location of image input folder
    :param folder: location to set
    :return: None
    """
    config_main.APPL_INPUT = config_main.IMAGE_INPUT
    config_main.APPL_INPUT_DIR = folder

    log_setup_info_to_console('IMAGE FOLDER INPUT:{}'.format(os.path.join(os.getcwd(), config_main.APPL_INPUT_DIR)))


def set_input_video(path: str) -> None:
    """
    Service that sets up the video path.
    :param path: location
    :return: None
    """
    config_main.APPL_INPUT = config_main.VIDEO_INPUT
    config_main.APPL_INPUT_VIDEO = path

    log_setup_info_to_console('VIDEO FOLDER INPUT:{}'.format(os.path.join(os.getcwd(), config_main.APPL_INPUT_DIR)))


def set_number_waves(waves: int) -> None:
    """
    Service that sets the number of waves to be active in parallel
    :param waves: number
    :return: None
    """
    config_main.APPL_NR_WAVES = waves

    log_setup_info_to_console('NUMBER OF WAVES IN PARALLEL TO RUN BY APPL: {}'.format(str(config_main.APPL_NR_WAVES)))


def set_input_camera_video(frames: int) -> None:
    """
    Service that sets up the camera video nr of frames that we want to capture
    :param frames: number of frames to capture
    :return: None
    """
    config_main.APPL_INPUT = config_main.CAMERA_INPUT
    config_main.APPL_NR_FRAMES_CAPTURE = frames


def configure_save_pictures(location: str = 'DEFAULT', job_name_in_port: bool = False, ports_to_save='ALL') -> None:
    """
    Service that sets up the port that we want to save
    :param location: where to save
    :param job_name_in_port: if we want the job name to saved in the picture
    :param ports_to_save: what ports to save
                          use ALL for saving all ports
    :return: None
    """
    # if this function is executed user wants to save pictures
    config_main.APPL_SAVE_PICT = True
    config_main.APPL_SAVE_JOB_NAME = job_name_in_port

    save_port_list = []

    if location is not 'DEFAULT':
        config_main.APPL_SAVE_LOCATION = location

    if ports_to_save is 'ALL':
        for port in created_port_list:
            if port[-1] is True:
                save_port_list.append(port[0])
    else:
        for el in ports_to_save:
            for port in created_port_list:
                if el == port[0]:
                    save_port_list.append(port[0])

    config_main.APPL_SAVE_PICT_LIST = save_port_list

    log_setup_info_to_console('IMAGE SAVE FOLDER:{}'.format(os.path.join(os.getcwd(), config_main.APPL_SAVE_LOCATION)))
    log_setup_info_to_console('PORTS TO SAVE:{}'.format(config_main.APPL_SAVE_PICT_LIST))
    log_setup_info_to_console('NR OF PORTS TO SAVE:{}'.format(len(config_main.APPL_SAVE_PICT_LIST)))


def configure_show_pictures(ports_to_show: list = 'ALL', time_to_show: int = 0, to_rotate: bool = False) -> None:
    """
    Service that sets up the port that we want to show
    :param ports_to_show: what ports to save
                          use ALL for saving all ports
    :param time_to_show: Time to show a picture
                         Use 0 for infinite show
    :param to_rotate: if we want to rotate images
    :return:
    """
    # if this function is executed user wants to save pictures
    config_main.APPL_SHOW_PICT = True
    config_main.APPL_SHOW_TIME = time_to_show
    save_port_list = []

    if ports_to_show is 'ALL':
        for port in created_port_list:
            if port[-1] is True:
                save_port_list.append(port[0])
    else:
        for el in ports_to_show:
            for port in created_port_list:
                if el == port[0]:
                    save_port_list.append(port[0])

    config_main.APPL_SHOW_LIST = list(set(save_port_list))

    if to_rotate:
        rotate_picture()

    log_setup_info_to_console('PORTS TO SHOW:{}'.format(config_main.APPL_SHOW_LIST))
    log_setup_info_to_console('PORTS TIME TO SHOW:{}'.format(config_main.APPL_SHOW_TIME))
    log_setup_info_to_console('NR OF PORTS TO SHOW:{}'.format(len(config_main.APPL_SHOW_LIST)))


def delete_folder_appl_out() -> None:
    """
    Service that deletes the content of out folder where the saved images are.
    :return: None
    """
    path = os.path.join(os.getcwd(), config_main.APPL_SAVE_LOCATION, '*')
    files = glob.glob(path)

    for f in files:
        shutil.rmtree(f, ignore_errors=True)

    log_setup_info_to_console('DELETED CONTENT OF: {}'.format(config_main.APPL_SAVE_LOCATION))


def create_list_ports_with_word(word: str) -> list:
    """
    Service for creating a list of ports with a certain word
    :param word: string to search for
    :return: None
    """
    port_list = []
    for port in created_port_list:
        if port[-1] is True and word in port[0]:
            port_list.append(port[0])

    return port_list


def create_list_ports_start_with_word(word: str) -> list:
    """
    Service for creating list of ports with a certain word
    :param word: string to search for
    :return: None
    """
    port_list = []
    for port in created_port_list:
        if port[-1] is True and port[0].startswith(word):
            port_list.append(port[0])

    return port_list


def define_output_extension(extension: str) -> None:
    """
    Service for setting extension of output files
    :param extension: extension to use
    :return: None
    """
    config_main.APPl_SAVE_PICT_EXTENSION = '.' + extension


def create_folder_from_list_ports(folder_name: str, list_port: list) -> None:
    """
    Copies ports from multiple output folders of ports into one
    :param folder_name: name of folder
    :param list_port: list of ports
    :return: None
    """
    if not os.path.exists(os.path.join(folder_name)):
        os.makedirs(os.path.join(folder_name))

    for port in list_port:
        list_img = [x for x in os.listdir(os.path.join(config_main.APPL_SAVE_LOCATION, port))]
        for file in list_img:
            src = os.path.join(config_main.APPL_SAVE_LOCATION, port, file)
            dst = os.path.join(folder_name, file)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    pass
