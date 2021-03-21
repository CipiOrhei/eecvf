from Utils.log_handler import log_to_console, log_end_of_wave
from Application.Frame.job import Job
from Application.Utils.parseJsonFile import FIELD_POSITION
# noinspection PyPep8Naming
import config_main as CONFIG

if CONFIG.CUDA_GPU is True:
    import tensorflow as tf

"""
Module handles the manipulations of jobs for the APPL block
"""


def job_creation(job_description: list) -> list:
    """
    Creates a list of job objects.
    :param job_description: a list of tuples that contain the necessary information for construction jobs
           See FIELD_POSITION in parseJsonFile module for details
    :return: list of jobs
    """
    job_list = []

    for job in job_description:
        job_list.append(Job(name=job[FIELD_POSITION.NAME], main_function=job[FIELD_POSITION.MAIN_FUNC],
                            init_function=job[FIELD_POSITION.INIT_FUNC], output_ports=job[FIELD_POSITION.OUTPUT_PORTS],
                            input_ports=job[FIELD_POSITION.INPUT_PORTS], init_func_param=job[FIELD_POSITION.INIT_FUNC_PARAM],
                            main_func_param=job[FIELD_POSITION.MAIN_FUNC_PARAM], waves=CONFIG.APPL_NR_WAVES))

        log_to_console(job_list[-1].get_echo())

    return job_list


def init_jobs(list_jobs: list) -> None:
    """
    Initialises the list of jobs
    :param list_jobs:
    :return: None
    """
    for job in list_jobs:
        job.init()

    log_end_of_wave()


def terminate_jobs(list_jobs: list) -> None:
    """
    Terminates the jobs in the list
    :param list_jobs:
    :return: None
    """
    for job in list_jobs:
        job.terminate()
        
    if CONFIG.CUDA_GPU is True:
        tf.keras.backend.clear_session()


def log_to_console_avg_time(job_list: list) -> None:
    """
    Logs to console average time of job run
    :param job_list:
    :return: None
    """
    log_to_console("Average time of jobs:")

    for job in job_list:
        log_to_console("JOB : {job:150s} AVERAGE TIME[ms]: {time:10.4f}".format(job=job.__name__, time=job.get_average_time()))


if __name__ == "__main__":
    pass
