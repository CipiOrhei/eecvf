import numpy as np

from Application.Frame.global_variables import JobInitStateReturn

from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_file, log_error_to_console


def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def main_func(param_list: list = None) -> bool:
    """
    Main function for label list correction.
    :param param_list: Param needed to respect the following list:
                       [port_in image: image input, wave_offset, input_class_list, output_class_list, output_image]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_CLASS_INPUT = 2
    # noinspection PyPep8Naming
    PORT_IN_CLASS_OUTPUT = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 4

    if len(param_list) != 5:
        log_error_to_console("CLASS CORRELATION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_out_img = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                tmp = np.zeros(port_in.arr.shape)

                for el in range(len(param_list[PORT_IN_CLASS_INPUT])):
                    tmp += (port_in.arr == param_list[PORT_IN_CLASS_INPUT][el]) * param_list[PORT_IN_CLASS_OUTPUT][el]

                port_out_img.arr[:] = tmp.astype(np.uint8)
                port_out_img.set_valid()

            except BaseException as error:
                log_to_file('')
                log_error_to_console("CLASS CORRELATION JOB NOK: ", str(error))
                pass
        else:
            log_to_file('')
            return False

        return True


if __name__ == "__main__":
    pass
