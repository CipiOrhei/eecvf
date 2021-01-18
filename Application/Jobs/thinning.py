# noinspection PyPackageRequirements
import thinning
# Do not delete used indirectly
# noinspection PyUnresolvedReferences
from Application.Frame import transferJobPorts
from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console


def init_func() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def main_guo_hall_func(port_list: list = None) -> bool:
    """
    Guo - Hall explained in "Parallel thinning with two sub-iteration algorithms" by Zicheng Guo and Richard Hall

    :param port_list: Param needed list of port names [input1, kernel_size, sigma, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_OUT_POS = 2

    # check if param OK
    if len(port_list) != 3:
        log_error_to_console("GUA HALL THINNING JOB MAIN FUNCTION PARAM NOK", str(len(port_list)))
        return False
    else:
        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True:
            try:
                p_out.arr[:] = thinning.guo_hall_thinning(p_in.arr.copy())
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("GUA HALL THINNING JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


if __name__ == "__main__":
    pass
