import json
import os

# noinspection PyPep8Naming
import config_main as CONFIG
from Application.Frame.transferJobPorts import create_ports_dict
from Utils.log_handler import log_job_to_config, log_setup_info_to_console, verbose_log_to_console

"""
Module handles the processing of desired jobs into json config file that is used by APPL layer for execution
"""

jobs_dict = []
# list of (name, save, show, is_image) -> not to search all the dictionary for them
created_port_list = []
# Process level default value
PROCESS_DEFAULT_VALUE = -1


def process_ports_list(ports_list: list):
    """
    :param ports_list: list of ports
                       [{'port name': 'RAW_L0', 'port size': 'L0_SIZE_RGB', 'port type': 'B'}]
    :return: A list only with the port name
            'RAW_L0, '
    """
    string_to_return = ''

    if ports_list is None:
        string_to_return = 'None'
    else:
        for i in range(len(ports_list)):
            string_to_return += ports_list[i]['port name'] + ", "

    return string_to_return


def create_dictionary_element(job_module: str, job_name: str, init_func_name: str, main_func_name: str,
                              main_func_param: list = None, output_ports: list = None, init_func_param: list = None,
                              input_ports: list = None, max_wave: int = 0):
    """
    Creates a dictionary element for a job.
    :param job_module: name of the python module where the job functions are
    :param job_name: what name do you want your job to have
    :param input_ports: list of input port your job needs
    :param init_func_name: name of init function from job module
    :param init_func_param: list of init function param
    :param main_func_name: maine function name from job module
    :param main_func_param: list of maine function param
    :param output_ports: list of output ports
    :param max_wave: port wave offset. If 0 it is in current wave.
    :return: dictionary for the job
    """
    assert max_wave < CONFIG.APPL_NR_WAVES, 'CONFIG ERROR: PLEASE CHECK NUMBER OF WAVES'

    dict_element = dict()

    dict_element['processing level'] = PROCESS_DEFAULT_VALUE
    dict_element['active'] = True
    dict_element['package'] = 'Jobs'
    dict_element['module'] = job_module
    dict_element['name'] = job_name

    input_port_list = []
    if input_ports is not None:
        for port in input_ports:
            input_port_list.append({'port name': port})
    else:
        input_port_list = None

    dict_element['input ports'] = input_port_list

    dict_element['init function'] = init_func_name

    init_param_list = []
    if init_func_param is not None:
        for port in init_func_param:
            init_param_list.append({'param': port})
    else:
        init_param_list = None

    dict_element['init function parameters'] = init_param_list

    dict_element['main function'] = main_func_name

    main_param_list = []
    if main_func_param is not None:
        for port in main_func_param:
            main_param_list.append({'param': port})
    else:
        main_param_list = None

    dict_element['main function parameters'] = main_param_list

    output_port_list = []
    if output_ports is not None:
        for port in output_ports:
            output_port_list.append({'port name': port[0], 'port size': port[1], 'port type': port[2], 'is image': port[3]})
            # add port to internal buffer
            created_port_list.append((port[0], False, False, port[3]))
    else:
        output_port_list = None

    dict_element['output ports'] = output_port_list

    log_job_to_config(dict_element['name'], process_ports_list(dict_element['input ports']),
                      process_ports_list(dict_element['output ports']))

    return dict_element


def find_duplicates_in_jobs(verbose: bool = False):
    """
    Function that eliminates the identical jobs from list
    :param verbose: if we want debug
    :return: None
    """
    if verbose:
        verbose_log_to_console('FIND DUPLICATES JOBS FUNCTION: Job list at start')
        for job in jobs_dict:
            text = 'Job name: {name:40s} level: {lvl:4d} inputs: {a:60s} output: {b:60s} main_func: {m:30s} init_func:{n}'.format(
                name=job['name'], lvl=job['processing level'], a=process_ports_list(job['input ports']),
                b=process_ports_list(job['output ports']),
                m=job['main function'], n=job['init function'])
            verbose_log_to_console(text)

    # find duplicates in each level
    for level in range(1, jobs_dict[-1]['processing level'] + 1, 1):
        for index in range(len(jobs_dict)):
            if jobs_dict[index]['processing level'] == level and jobs_dict[index]['active'] is True:
                for other_index in range(index + 1, len(jobs_dict), 1):
                    for key in jobs_dict[index].keys():
                        if jobs_dict[index][key] != jobs_dict[other_index][key]:
                            break
                    else:
                        jobs_dict[other_index]['active'] = False

    # delete identical jobs
    while True:
        for index in range(len(jobs_dict)):
            if not jobs_dict[index]['active']:
                log_setup_info_to_console(jobs_dict[index]['name'] + ' DUPLICATED JOB -> JOB DELETED')
                del jobs_dict[index]
                break
        else:
            break

    if verbose:
        verbose_log_to_console('FIND DUPLICATES JOBS FUNCTION: Job list at start')
        for job in jobs_dict:
            text = 'Job name: {name:40s} level: {lvl:4d} inputs: {a:60s} output: {b:60s} main_func: {m:30s} init_func:{n}'.format(
                name=job['name'], lvl=job['processing level'], a=process_ports_list(job['input ports']),
                b=process_ports_list(job['output ports']),
                m=job['main function'], n=job['init function'])
            verbose_log_to_console(text)


def sort_jobs_to_avoid_missing_inputs(verbose: bool = False):
    """
    :param verbose: if we want debug
    Resets process level of jobs to order them in input availability order
    First job will allays pe set the image retrieval job
    Second job will be the pyramid level job to set up the possibility of parallel run in application
    Will disable jobs without inputs.
    :return: None
    """
    process_index = 0

    if verbose:
        verbose_log_to_console('SORT JOBS FUNCTION: After resetting process level')
        for job in jobs_dict:
            text = 'Job name {name:40s} has process level: {lvl:4d} inputs: {a:100s} output: {b}'.format(
                name=job['name'], lvl=job['processing level'], a=str(job['input ports']), b=str(job['output ports']))
            verbose_log_to_console(text)

    # find begging of pipeline
    for job in jobs_dict:
        # get image job should be the first
        if job['input ports'] is None:
            job['processing level'] = process_index
            process_index += 1

    # loop continuous for jobs.
    last_jobs_without_process_counter = 0
    while True:
        jobs_without_process = []
        jobs_with_process = []

        # create 2 list with processed jobs and unprocessed
        for position in range(len(jobs_dict)):
            if jobs_dict[position]['processing level'] == PROCESS_DEFAULT_VALUE:
                jobs_without_process.append([position, [jobs_dict[position]['input ports'][p]['port name'] for p in
                                                        range(len(jobs_dict[position]['input ports']))]])
            else:
                jobs_with_process.append([position, [jobs_dict[position]['output ports'][p]['port name'] for p in
                                                     range(len(jobs_dict[position]['output ports']))]])

        # break loop if all jobs were sorted
        if len(jobs_without_process) == 0:
            break

        # break loop if we can't sort the remaining jobs
        if last_jobs_without_process_counter != len(jobs_without_process):
            last_jobs_without_process_counter = len(jobs_without_process)
        else:
            break

        # for each unresolved job we search if the necessary inputs are found in a resolved job
        for unresolved_job in jobs_without_process:
            inputs_found = []
            for resolved_job in jobs_with_process:
                for input_job in unresolved_job[-1]:
                    if input_job in resolved_job[-1] and input_job not in inputs_found:
                        inputs_found.append(input_job)

            if inputs_found.sort() == unresolved_job[-1].sort():
                jobs_dict[unresolved_job[0]]['processing level'] = process_index

        process_index += 1

    if verbose:
        verbose_log_to_console('SORT JOBS FUNCTION: After loop for process level')
        for job in jobs_dict:
            text = 'Job name {name:40s} has process level: {lvl:4d} inputs: {a:100s} output: {b}'.format(
                name=job['name'], lvl=job['processing level'], a=str(job['input ports']), b=str(job['output ports']))
            verbose_log_to_console(text)

    # set to invalid jobs without inputs
    for job in jobs_dict:
        if job['processing level'] == PROCESS_DEFAULT_VALUE:
            job['active'] = False
            log_setup_info_to_console(job['name'] + ' INPUTS MISSING -> JOB DISABLED')


def create_config_file(verbose: bool = False):
    """
    :param verbose: if we want debug
    Creates config file for application to run on
    :param verbose: if we want debug
    :return: None
    """
    global jobs_dict

    if not os.path.exists(os.path.join(os.getcwd(), CONFIG.JSON_FILE_LOCATION)):
        os.makedirs(CONFIG.JSON_FILE_LOCATION)

    file = open(os.path.join(os.getcwd(), CONFIG.JSON_FILE_LOCATION, CONFIG.JSON_FILE_NAME + '.json'), 'w')

    if verbose:
        verbose_log_to_console('Json file location: ' + str(file))

    log_setup_info_to_console('NUMBER OF JOBS TRIGGERED BY USER: {}'.format(len(jobs_dict)))

    if len(jobs_dict) > 0:
        sort_jobs_to_avoid_missing_inputs(verbose)
        data_list = sorted(jobs_dict, key=lambda x: x['processing level'])
        find_duplicates_in_jobs(verbose)

        for el in data_list:
            del el['processing level']

        data_to_write = json.dumps(data_list, indent=2)

        if verbose:
            verbose_log_to_console('Data that will be writen in json file: ')
            verbose_log_to_console(data_to_write)

        log_setup_info_to_console('NUMBER OF JOBS ADDED TO JSON TO RUN BY APPL: {}'.format(len(jobs_dict)))
        log_setup_info_to_console('NUMBER OF PORTS ADDED TO JSON TO RUN BY APPL: {}'.format(len(created_port_list)))

        file.write(data_to_write)
        file.close()

    CONFIG.APPL_INPUT_JOB_LIST = file.buffer.name
    create_ports_dict(CONFIG.APPL_NR_WAVES)
    jobs_dict.clear()


if __name__ == "__main__":
    pass
