import json
# Do not delete used indirectly
# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import Application.Jobs
from Application.Frame.global_variables import global_var_handler
from Utils.log_handler import log_error_to_console

"""
Module handles the json file parsing for the APPL block
"""

json_fields = []

check_port_size = lambda value: int(value) if value.isdigit() is True else global_var_handler.get_size_equivalence(value)


class FIELD_POSITION:
    """
    description of constructor param positions
    """
    NAME = 0
    INPUT_PORTS = 1
    MAIN_FUNC = 3
    INIT_FUNC = 2
    OUTPUT_PORTS = 4
    MAIN_FUNC_PARAM = 6
    INIT_FUNC_PARAM = 5
    NR_FIELDS = 7


def convert_json_to_job(element: dict) -> list:
    """
    Converts information from json file regarding a job to param necessary to construct a job
    :param element: job from json file
    :return: list of job details for constructor
    """
    job_entry = list()

    # get job name
    job_entry.append(element['name'])
    # get jobs input ports
    if element['input ports'] is not None:
        job_entry.append([elm['port name'] for elm in element['input ports']])
    else:
        job_entry.append(None)
    # get init function
    job_entry.append(eval('Application.' + element['package'] + '.' + element['module'] + '.' + element['init function']))
    # get main function
    job_entry.append(eval('Application.' + element['package'] + '.' + element['module'] + '.' + element['main function']))
    # get output ports
    if element['output ports'] is not None:
        job_entry.append(
            [(elm['port name'], check_port_size(elm['port size']), elm['port type'], elm['is image']) for elm in element['output ports']])
    else:
        job_entry.append(None)
    # get init func param
    if element['init function parameters'] is not None:
        job_entry.append([(elm['param']) for elm in element['init function parameters']])
    else:
        job_entry.append(None)
    # get main func param
    if element['main function parameters'] is not None:
        job_entry.append([(elm['param']) for elm in element['main function parameters']])
    else:
        job_entry.append(None)

    return job_entry


def get_json_data(file: str) -> list:
    """
    Retrieving data from configuration json file
    :param file: json file
    :return: a list of jobs
    """
    job_list = []

    try:
        data = json.load(open(file))
        # get fields from json files
        global json_fields
        for key in data[0].keys():
            json_fields.append(key)
        # TODO try to find better solution than hardcode 3
        # verify fields didn't change
        assert FIELD_POSITION.NR_FIELDS == (len(json_fields) - 3), "Check json file and job fields!!!"
        # create list of jobs
        for element in data:
            if element['active'] is not False:
                job_list.append(tuple(convert_json_to_job(element)))

        json_fields.clear()
    except BaseException as error:
        log_error_to_console("JSON JOB CONFIG FIELDS NOK: ", str(error))
        pass

    return job_list


def get_jobs(json_file: str = None) -> list:
    """
    Get's the json file and parses it.
    :param json_file: location of json file
    :return: list of jobs information for constructing them
    """
    try:
        # noinspection PyTypeChecker
        return get_json_data(file=json_file)
    except BaseException as error:
        log_error_to_console('JSON JOB CONFIG LOCATION NOK ' + str(json_file), str(error))
        return []


if __name__ == "__main__":
    pass
