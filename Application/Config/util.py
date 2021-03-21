# noinspection PyPackageRequirements
from config_main import PYRAMID_LEVEL
import sys

"""
Module handles misc helper functions
"""


def transform_port_name_lvl(name: str, lvl: PYRAMID_LEVEL):
    """
    Function for creating name of port for manipulation.
    :param name: name of port
    :param lvl: pyramid level of port
    :return: port name
    """
    return name + '_' + str(lvl)


def transform_port_size_lvl(lvl: PYRAMID_LEVEL, rgb: bool):
    """
    Function for creating size of ports.
    :param lvl: pyramid level of port
    :param rgb: is port rgb or greyscale
    :return: string of port size
    """
    if rgb is False:
        return str(lvl) + '_SIZE'
    else:
        return str(lvl) + '_SIZE_RGB'


def job_name_create(action: str, input_list: list = None, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: list = 0, **kwargs):
    """
    Function for job name.
    :param action: string representing the action done by the job
    :param input_list: list of strings representing the input ports
    :param wave_offset: list of ints representing the wave of each input port
    :param level: pyramid level on which the job will take place
    :param kwargs: list of parameter for job configuration( use example : Var=5)
    :return: job name
    """
    string = str(action)

    if input_list is not None:
        if 'Add' in action:
            string += ' to '
        else:
            string += ' of '

        if len(input_list) == 1:
            string += input_list[0] + ' W-' + str(wave_offset[0])
        else:
            for i in range(len(input_list)):
                if i != 0:
                    string += ' and '
                string += input_list[i] + ' W-' + str(wave_offset[i])

    if len(kwargs) is not 0:
        string += ' with '
        for i in range(len(kwargs)):
            if i != 0:
                string += ' '
            string += str(list(kwargs.keys())[i]) + '=' + str(kwargs[list(kwargs.keys())[i]])

    string += ' on ' + str(level)

    return string


def get_module_name_from_file(file: str):
    """
    Function for parsing from __file__ build in variable to string of module name.
    :param file: __file__ build in variable
    :return: string of module name.
    """
    if '\\' in file:
        return (file.split('\\')[-1]).split('.py')[0]
    elif '/' in file:
        return (file.split('/')[-1]).split('.py')[0]
