# noinspection PyPackageRequirements
import cv2
# Do not delete used indirectly
# noinspection PyUnresolvedReferences
from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console

"""
Module handles the pyramid transition jobs in the APPL block.
"""


def init_func() -> JobInitStateReturn:
    """
    Init function for the job
    :port_list Number of lvl to go
    :return: INIT or NOT_INIT state for the job
    """

    return JobInitStateReturn(True)


def main_func_down(port_list: list = None) -> bool:
    """
    Main function for pyramid level downsizing using cv2 function job.
    :param port_list: Param needed list of port names
                      [input1, wave_offset,
                      output1, output2, output3, output4...]
                  List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # check if param OK
    if len(port_list) < 3:
        log_error_to_console("PYRAMID REDUCE PARAM NOK", str(len(port_list)))
        return False
    else:
        for el in range(2, len(port_list)):
            if el is 2 and port_list[0] is not 0:
                p_in = get_port_from_wave(name=port_list[el - 1], wave_offset=port_list[0])
            else:
                p_in = get_port_from_wave(name=port_list[el - 1], wave_offset=0)
            p_out = get_port_from_wave(name=port_list[el])

            if p_in.is_valid() is True:
                try:
                    level = p_out.name.split('_')[-1]
                    p_out.arr[:] = cv2.pyrDown(src=p_in.arr,
                                               dstsize=(eval('global_var_handler.WIDTH_' + level), eval('global_var_handler.HEIGHT_' + level)))
                    p_out.set_valid()
                except BaseException as error:
                    log_error_to_console("PYRAMID REDUCE JOB NOK: ", str(error))
                    break
            else:
                return False
        return True


def main_func_up(port_list: list = None) -> bool:
    """
    Main function for pyramid level upsizing using cv2 function job.
    :param port_list: Param needed list of port names
                     [wave_offset, input1,
                      output1, output2, output3, output4...]
                  List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # check if param OK
    if len(port_list) < 3:
        log_error_to_console("PYRAMID EXPAND PARAM NOK", str(len(port_list)))
        return False
    else:
        for el in range(2, len(port_list)):
            if el is 2 and port_list[0] is not 0:
                p_in = get_port_from_wave(name=port_list[el - 1], wave_offset=port_list[0])
            else:
                p_in = get_port_from_wave(name=port_list[el - 1], wave_offset=0)
            p_out = get_port_from_wave(name=port_list[el])

            if p_in.is_valid() is True:
                try:
                    level = p_out.name.split('_')[-1]
                    p_out.arr[:] = cv2.pyrUp(src=p_in.arr,
                                             dstsize=(eval('global_var_handler.WIDTH_' + level), eval('global_var_handler.HEIGHT_' + level)))
                    p_out.set_valid()
                except BaseException as error:
                    log_error_to_console("PYRAMID EXPAND JOB NOK: ", str(error))
                    break
            else:
                return False

        return True


if __name__ == "__main__":
    pass
