# import what you need

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file

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
def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :return: INIT or NOT_INIT state for the job
    """
    log_to_file('DATA YOU NEED TO SAVE EVERY FRAME IN CSV')
    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################

# define a main function, function that will be executed at the begging of the wave
def main_func(param_list: list = None) -> bool:
    """
    Main function for {job} calculation job.
    :param param_list: Param needed to respect the following list:
                       [enumerate list]
    :return: True if the job executed OK.
    """
    # variables for position of param needed
    # ex:

    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    # VAL_NEEDED = 1
    # verify that the number of parameters are OK.
    if len(param_list) != 0:
        log_error_to_console("JOB_NAME JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS])
        # log to console information
        log_to_console('USE THIS ONLY FOR IMPORTANT MSG NOT JOB STATE, PORT STATE OR TIME')

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                pass
            except BaseException as error:
                log_error_to_console("JOB_NAME JOB NOK: ", str(error))
                pass
        else:
            return False

        return True

############################################################################################################################################
# Job create functions
############################################################################################################################################

def do_user_interface_job(port_input_name: str,
                          # specific parameters
                          param_value = 0,
                          port_img_output: str = None,
                          level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    User-interface of {job}
    Add details of job
    Please add paper from feature
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
        # one example
        # port_img_output = 'SIFT_KP_NF_{nf}_NO_{no}_CT_{ct}_ET_{et}_G_{g}_{INPUT}'.format(nf=number_features, no=number_octaves,
        #                                                                                 ct=contrast_threshold.__str__().replace('.', '_'),
        #                                                                                 et=edge_threshold, g=gaussian_sigma.__str__().replace('.', '_'),
        #                                                                                 INPUT=port_input_name)
        port_img_output = '{name}_{param}={value}_{Input}'.format(name='JOB_NAME',
                                                                  param='RELEVANT_PARAMS',
                                                                  value=param_value.__str__().replace('.', '_'),
                                                                  Input=port_input_name)

    # size can be custom as needed
    port_img_output_name = transform_port_name_lvl(name=port_img_output, lvl=level)
    # new_size = (new_height, new_width)
    # level = PYRAMID_LEVEL.add_level(size=new_size)
    # port_des_output_name_size = '({nr_kp}, 128)'.format(nr_kp=number_features)
    port_img_output_name_size = transform_port_size_lvl(lvl=level, rgb=True)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, param_value, port_img_output_name]
    # ['port name', 'port size', 'port type', 'is image']
    output_port_list = [(port_img_output_name, port_img_output_name_size, 'B', True)]

    job_name = job_name_create(action='JOB_NAME', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_img_output


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
