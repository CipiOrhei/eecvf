import numpy as np

from Application.Frame.global_variables import JobInitStateReturn

from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_file, log_error_to_console

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL, FILTERS
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file


############################################################################################################################################
# Init functions
############################################################################################################################################

def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################


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
                tmp = np.zeros(port_out_img.arr.shape)

                if len(port_in.arr.shape) == 3:
                    for el in range(len(param_list[PORT_IN_CLASS_INPUT])):
                        t1 = (port_in.arr[:, :, 0] == param_list[PORT_IN_CLASS_INPUT][el][0])
                        t2 = (port_in.arr[:, :, 1] == param_list[PORT_IN_CLASS_INPUT][el][1])
                        t3 = (port_in.arr[:, :, 2] == param_list[PORT_IN_CLASS_INPUT][el][2])

                        tmp += np.bitwise_and(t1, np.bitwise_and(t2, t3)) * param_list[PORT_IN_CLASS_OUTPUT][el]
                elif len(port_in.arr.shape) == 2 and len(port_out_img.arr.shape) == 2:
                    for el in range(len(param_list[PORT_IN_CLASS_INPUT])):
                        tmp += (port_in.arr == param_list[PORT_IN_CLASS_INPUT][el]) * param_list[PORT_IN_CLASS_OUTPUT][el]

                elif len(port_in.arr.shape) == 2:
                    for el in range(len(param_list[PORT_IN_CLASS_INPUT])):
                        t = tmp.copy()
                        t[:,:,0] = (port_in.arr == param_list[PORT_IN_CLASS_INPUT][el]) * param_list[PORT_IN_CLASS_OUTPUT][el][0]
                        t[:,:,1] = (port_in.arr == param_list[PORT_IN_CLASS_INPUT][el]) * param_list[PORT_IN_CLASS_OUTPUT][el][1]
                        t[:,:,2] = (port_in.arr == param_list[PORT_IN_CLASS_INPUT][el]) * param_list[PORT_IN_CLASS_OUTPUT][el][2]

                        tmp += t

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


############################################################################################################################################
# Job create functions
############################################################################################################################################

# TODO make RGB->RGB work
def do_class_correlation(port_input_name: str,
                         class_list_in: list, class_list_out: list,
                         port_output_name: str = None,
                         level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> None:
    """
    Modify label values for an labeled image from the class_list_in to class_list_out labels.
    :param port_input_name: name of input port
    :param class_list_in: list of input classes
    :param class_list_out: list of output classes
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'CLASS_CORRELATION_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)

    rgb_output = True
    if isinstance(class_list_out[0], int):
        rgb_output = False

        # tmp_img = np.rint(0.2989 * tmp_img[:,2] + 0.5870 * tmp_img[:,1] + 0.1140 * tmp_img[:,0])
        # class_list_in = tmp_img.tolist()

    output_port_size = transform_port_size_lvl(lvl=level, rgb=rgb_output)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, class_list_in, class_list_out, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Class correlation', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               LIST_IN=class_list_in, LIST_OUT=class_list_out)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)



if __name__ == "__main__":
    pass
