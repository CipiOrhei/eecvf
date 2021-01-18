# import what you need

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console

"""
Module handles DESCRIPTION OF THE MODULE jobs for the APPL block.
"""


# define a init function, function that will be executed at the begging of the wave
def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :return: INIT or NOT_INIT state for the job
    """
    log_to_file('DATA YOU NEED TO SAVE EVERY FRAME IN CSV')
    return JobInitStateReturn(True)


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


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
