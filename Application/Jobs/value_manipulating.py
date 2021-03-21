# import what you need

import numpy as np

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, job_name_create, get_module_name_from_file

"""
Module handles value manipulation jobs for the APPL block.
"""


# define a init function, function that will be executed at the begging of the wave
def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :return: INIT or NOT_INIT state for the job
    """
    log_to_file('Value manipulation output')
    return JobInitStateReturn(True)


# define a main function, function that will be executed at the begging of the wave
def main_func(param_list: list = None) -> bool:
    """
    Main function for value manipulation calculation job.
    :param param_list: Param needed to respect the following list:
                       [list of terms, list of wave of terms, list of operations, output port]
    :return: True if the job executed OK.
    """
    # variables for position of param needed
    # ex:

    # noinspection PyPep8Naming
    PORT_IN_LIST_TERMS_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_LIST_WAVE_POS = 1
    # noinspection PyPep8Naming
    PORT_OPERATIONS_POS = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3
    # verify that the number of parameters are OK.
    if len(param_list) != 4:
        log_error_to_console("VALUE MANIPULATION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        term_list = []
        ports_valid = True
        # get needed ports
        for p_idx in range(len(param_list[PORT_IN_LIST_TERMS_POS])):
            if isinstance(param_list[PORT_IN_LIST_TERMS_POS][p_idx], str):
                port = get_port_from_wave(name=param_list[PORT_IN_LIST_TERMS_POS][p_idx],
                                          wave_offset=param_list[PORT_IN_LIST_WAVE_POS][p_idx])
                term = port.arr
                if isinstance(term, (np.ndarray, np.generic)):
                    for el in term:
                        term_list.append(el)
                ports_valid = ports_valid and port.is_valid()
            else:
                term_list.append(param_list[PORT_IN_LIST_TERMS_POS][p_idx])

        p_out = get_port_from_wave(name=param_list[PORT_OUT_POS])

        # check if port's you want to use are valid
        if ports_valid is True:
            try:
                if len(param_list[PORT_IN_LIST_TERMS_POS]) - 1 == len(param_list[PORT_OPERATIONS_POS]):

                    txt_operation = str(term_list[0])

                    for idx in range(len(param_list[PORT_OPERATIONS_POS])):
                        txt_operation += str(param_list[PORT_OPERATIONS_POS][idx])
                        txt_operation += str(term_list[idx + 1])

                    p_out.arr[:] = eval(txt_operation)
                    p_out.set_valid()

                    log_to_file(p_out.arr[0].__str__())
                else:
                    log_to_file('')
                    log_error_to_console("VALUE MANIPULATION JOB NOK: ", "LESS OPERATIONS THAN NEEDED")
            except BaseException as error:
                log_to_file('')
                log_error_to_console("VALUE MANIPULATION JOB NOK: ", str(error))
                pass
        else:
            log_to_file('')
            return False

        return True


############################################################################################################################################
# Job create functions
############################################################################################################################################


def do_value_manipulation_job(terms_input_list: list, port_input_wave_list: list, port_input_level_list: list,
                              operation_list: list, port_output: str,
                              level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Job for mathematical manipulation of ports that contain values.
    If wave of level not needed in term, because it is not a port use '' but not leave empty.
    :param terms_input_list: list of ports to use
    :param port_input_wave_list: list of wave of ports to use
    :param port_input_level_list: list of level of ports to use
    :param operation_list: list of operations to use. ! Be aware that the list of operation must be n-1 compared with list of port_input_list
    :param port_output: output port
    :param level: pyramid level to calculate at
    :return: output image port name
    """
    if len(terms_input_list) != len(port_input_wave_list) != len(port_input_level_list):
        log_error_to_console("MANIPULATION_VALUE_JOB NOK: ", "LIST INPUT SIZE DO NOT MATCH")

    input_port_list = []
    main_param_input_list = []

    for idx_input in range(len(terms_input_list)):
        if isinstance(terms_input_list[idx_input], str):
            input_port_name = transform_port_name_lvl(name=terms_input_list[idx_input], lvl=port_input_level_list[idx_input])
            input_port_list.append(input_port_name)
            main_param_input_list.append(input_port_name)
        else:
            main_param_input_list.append(terms_input_list[idx_input])

    output_port_name = transform_port_name_lvl(name=port_output, lvl=level)

    main_func_list = [main_param_input_list, port_input_wave_list, operation_list, output_port_name]
    output_port_list = [(output_port_name, '1', 'B', False)]

    job_name = job_name_create(action='Value manipulation for port_output')
    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
