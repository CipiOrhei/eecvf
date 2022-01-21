import os
import config_main

"""
Module handles the logging for the EECVF
"""

# if we run from Application folder
if 'Application' in os.getcwd():
    config_main.LOG_KPI_FILE = '../' + config_main.LOG_KPI_FILE
    config_main.LOG_FILE = '../' + config_main.LOG_FILE

if not os.path.exists(config_main.LOG_KPI_FILE.split('/')[0]):
    os.makedirs('Logs')

file_KPI = open(file=config_main.LOG_KPI_FILE, mode='w')
file_log = open(file=config_main.LOG_FILE, mode='w')

# global variable to hold the log information
WAVE_LOG = ''
# global flag if error in wave
ERROR = False


def reopen_files():
    global file_KPI
    global file_log
    # delete any custom ports
    config_main.PYRAMID_LEVEL.delete_levels_add_runtime()
    # reopen files for debug
    file_KPI = open(file=config_main.LOG_KPI_FILE, mode='a')
    file_log = open(file=config_main.LOG_FILE, mode='a')

    import gc
    gc.collect(generation=2)


def is_error() -> None:
    """
    Set's a error in the log file so you can see that the specific wave had an error
    :return: None
    """
    global ERROR
    ERROR = True


def log_to_file(text: str) -> None:
    """
    Log data to log file
    :param text: what data to log
    :return: None
    """
    global WAVE_LOG
    WAVE_LOG += text + ','


def log_end_of_wave() -> None:
    """
    Logs data at end of wave
    :return:
    """
    global WAVE_LOG, ERROR
    file_KPI.write(WAVE_LOG + str(ERROR) + '\n')
    ERROR = False
    WAVE_LOG = ''


def log_to_console(text: str) -> None:
    """
    Logs data on console
    :param text: What to print and write in console log
    :return: None
    """
    text = 'APPLICATION: {}'.format(text)
    file_log.write(text + '\n')
    print(text)


def verbose_log_to_console(text: str) -> None:
    """
    Logs data on console
    :param text: What to print and write in console log
    :return: None
    """
    text = 'VERBOSE: ' + text
    file_log.write(text + '\n')
    print(text)


def log_setup_info_to_console(text: str) -> None:
    """
    Logs data on console
    :param text: What to print and write in console log
    :return: None
    """
    text = '### {} ###'.format(text)
    file_log.write(text + '\n')
    print(text)


def log_benchmark_info_to_console(text: str) -> None:
    """
    Logs data on console
    :param text: What to print and write in console log
    :return: None
    """
    text = 'BENCHMARK: {}'.format(text)
    file_log.write(text + '\n')
    print(text)


def log_ml_info_to_console(text: str) -> None:
    """
    Logs data on console
    :param text: What to print and write in console log
    :return: None
    """
    text = 'ML: {}'.format(text)
    file_log.write(text + '\n')
    print(text)


def log_util_info_to_console(text: str) -> None:
    """
    Logs data on console
    :param text: What to print and write in console log
    :return: None
    """
    text = 'DATA_PROCESSING: {}'.format(text)
    file_log.write(text + '\n')
    print(text)


def log_job_to_config(name: str, input_ports: str, output_ports: str) -> None:
    """
    Logs data on console
    :param name: job name to log
    :param input_ports: input ports to log
    :param output_ports: output ports to log
    :return: None
    """
    text = '### \nJOB   : {name:100s} \nINPUT : {input:155} \nOUTPUT: {output:155} \n###'. \
        format(name=name, input=input_ports, output=output_ports)

    file_log.write(text + '\n')
    print(text)


def log_error_to_console(detail_text: str = None, detail_msg: str = None) -> None:
    """
    Logs error data on console
    :param detail_text: Data about the error
    :param detail_msg: Msg from python error
    :return: None
    """
    text = 'ERROR: {error_detail}: {error_msg}'.format(error_detail=detail_text, error_msg=detail_msg)
    file_log.write(text + '\n')
    print(text)


def close_files():
    """
    Closes file resources
    :return: None
    """
    file_KPI.close()
    file_log.close()


if __name__ == "__main__":
    pass
