# import what you need
import numpy as np

from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
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

dict_store_data = dict()

def read_csv_file(csv_file_location):
    file = open(csv_file_location, "r")
    # get dictionary fields from csv header
    fields = file.readline().split(',')
    # eliminate new line character
    fields[-1] = fields[-1].split('\n')[0]
    d = dict()

    for line in file.readlines():
        data = line.split(',')
        new_obj = dict()

        for idx_field in range(len(fields)):
            new_obj[fields[idx_field]] = data[idx_field].replace('\n','')

        d[data[0]]=new_obj
    file.close()

    return d


# define a init function, function that will be executed at the begging of the wave
def init_func_csv_tmbud_file(param_list) -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :param param_list: Param needed list of port names [location_file]
    :return: INIT or NOT_INIT state for the job
    """

    if param_list[0] not in dict_store_data.keys():
        dict_store_data['TMBuD_csv'] = read_csv_file(param_list[0])

    log_to_file('TMBuD ' + param_list[1])

    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################

# define a main function, function that will be executed at the begging of the wave
def main_func_tmbud_csv(param_list: list = None) -> bool:
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
    PORT_OUT_POS = 1
    # verify that the number of parameters are OK.
    if len(param_list) != 2:
        log_error_to_console("GET DATA FROM TMBuD CSV JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_out = get_port_from_wave(name=param_list[PORT_OUT_POS])

        frame = global_var_handler.PICT_NAME[:-4]
        # check if port's you want to use are valid
        if frame in list(dict_store_data['TMBuD_csv'].keys()) and param_list[PORT_IN_POS] in list(dict_store_data['TMBuD_csv'][frame].keys()):
            # try:
            if True:
                tmp = dict_store_data['TMBuD_csv'][frame][param_list[PORT_IN_POS]]
                # port_out.arr[:] = (tmp.split(';')[0],tmp.split(';')[1])
                port_out.arr = np.array([tmp])
                log_to_file(tmp)
                port_out.set_valid()
            # except BaseException as error:
            #     log_error_to_console("GET DATA FROM TMBuD CSV JOB NOK: ", str(error))
            #     pass
        else:
            log_to_file('')
            return False

        return True

############################################################################################################################################
# Job create functions
############################################################################################################################################

def do_get_data_TMBuD_csv_job(csv_field: str, file_location: str, port_img_output,
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
    # size can be custom as needed
    port_img_output_name = transform_port_name_lvl(name=port_img_output, lvl=level)


    input_port_list = None
    init_param_list = [file_location, csv_field]
    main_func_list = [csv_field, port_img_output_name]
    output_port_list = [(port_img_output_name, '(1, 50)', 'u', False)]

    job_name = job_name_create(action='Get TMBuD csv data', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_csv_tmbud_file', init_func_param=init_param_list,
                                  main_func_name='main_func_tmbud_csv',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_img_output


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
