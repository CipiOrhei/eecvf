# noinspection PyPackageRequirements
import cv2
import numpy as np

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console

"""
Module handles first order magnitude derivatives directional edge detection image jobs for the APPL block.
"""


def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def main_func_cross(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in_name_gn: str, port_in_name_gnw: str, port_in_name_gw: str, port_in_name_gsw: str,
                        port_in_name_gs: str, port_in_name_gse: str, port_in_name_ge: str, port_in_name_gne: str,
                        port_out_img name: str, port_out_ang: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    INPUT_G1_POS = 0
    # noinspection PyPep8Naming
    INPUT_G2_POS = 1
    # noinspection PyPep8Naming
    INPUT_G3_POS = 2
    # noinspection PyPep8Naming
    INPUT_G4_POS = 3
    # noinspection PyPep8Naming
    INPUT_G5_POS = 4
    # noinspection PyPep8Naming
    INPUT_G6_POS = 5
    # noinspection PyPep8Naming
    INPUT_G7_POS = 6
    # noinspection PyPep8Naming
    INPUT_G8_POS = 7
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 8
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 9

    if len(param_list) != 10:
        log_error_to_console("8 DIRECTION CROSS EDGE DETECTION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in_g1 = get_port_from_wave(name=param_list[INPUT_G1_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_g2 = get_port_from_wave(name=param_list[INPUT_G2_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_g3 = get_port_from_wave(name=param_list[INPUT_G3_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_g4 = get_port_from_wave(name=param_list[INPUT_G4_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_g5 = get_port_from_wave(name=param_list[INPUT_G5_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_g6 = get_port_from_wave(name=param_list[INPUT_G6_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_g7 = get_port_from_wave(name=param_list[INPUT_G7_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_in_g8 = get_port_from_wave(name=param_list[INPUT_G8_POS], wave_offset=param_list[PORT_IN_WAVE])

        port_out_img = get_port_from_wave(name=param_list[PORT_OUT_IMG])

        if (port_in_g1.is_valid() and port_in_g2.is_valid() and port_in_g3.is_valid() and port_in_g4.is_valid() and
                port_in_g5.is_valid() and port_in_g6.is_valid() and port_in_g7.is_valid() and port_in_g8.is_valid()) is True:
            try:
                result = cv2.max(src1=port_in_g1.arr,
                                 src2=cv2.max(src1=port_in_g2.arr,
                                              src2=cv2.max(src1=port_in_g3.arr,
                                                           src2=cv2.max(src1=port_in_g4.arr,
                                                                        src2=cv2.max(src1=port_in_g5.arr,
                                                                                     src2=cv2.max(src1=port_in_g6.arr,
                                                                                                  src2=cv2.max(src1=port_in_g7.arr,
                                                                                                               src2=port_in_g8.arr)))))))
                port_out_img.arr[:] = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                port_out_img.set_valid()
            except BaseException as error:
                log_error_to_console("8 DIRECTION CROSS EDGE DETECTION JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_func_6_cross(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in_name_g1: str, port_in_name_g2: str, port_in_name_g3: str,
                        port_in_name_g4: str, port_in_name_g5: str, port_in_name_g6: str,
                        port_out_img name: str, port_out_ang: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    INPUT_G1_POS = 0
    # noinspection PyPep8Naming
    INPUT_G2_POS = 1
    # noinspection PyPep8Naming
    INPUT_G3_POS = 2
    # noinspection PyPep8Naming
    INPUT_G4_POS = 3
    # noinspection PyPep8Naming
    INPUT_G5_POS = 4
    # noinspection PyPep8Naming
    INPUT_G6_POS = 5
    # noinspection PyPep8Naming
    INPUT_WAVE_POS = 6
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 7

    if len(param_list) != 8:
        log_error_to_console("6 DIRECTION CROSS EDGE DETECTION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in_g1 = get_port_from_wave(name=param_list[INPUT_G1_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g2 = get_port_from_wave(name=param_list[INPUT_G2_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g3 = get_port_from_wave(name=param_list[INPUT_G3_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g4 = get_port_from_wave(name=param_list[INPUT_G4_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g5 = get_port_from_wave(name=param_list[INPUT_G5_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g6 = get_port_from_wave(name=param_list[INPUT_G6_POS], wave_offset=param_list[INPUT_WAVE_POS])

        port_out_img = get_port_from_wave(name=param_list[PORT_OUT_IMG])
        # TODO something wrong here
        if (port_in_g1.is_valid() and port_in_g2.is_valid() and port_in_g3.is_valid() and port_in_g4.is_valid()
                and port_in_g5.is_valid() and port_in_g6.is_valid()) is True:
            try:
                result = cv2.max(src1=port_in_g1.arr,
                                 src2=cv2.max(src1=port_in_g2.arr,
                                              src2=cv2.max(src1=port_in_g3.arr,
                                                           src2=cv2.max(src1=port_in_g4.arr,
                                                                        src2=cv2.max(src1=port_in_g5.arr,
                                                                                     src2=port_in_g6.arr)))))

                port_out_img.arr[:] = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                port_out_img.set_valid()
            except BaseException as error:
                log_error_to_console("6 DIRECTION CROSS EDGE DETECTION JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_func_frei_chen(param_list: list = None) -> bool:
    """
    Main function for gradient calculation job.
    :param param_list: Param needed to respect the following list:
                       [port_in_name_gn: str, port_in_name_gnw: str, port_in_name_gw: str, port_in_name_gsw: str,
                        port_in_name_gs: str, port_in_name_gse: str, port_in_name_ge: str, port_in_name_gne: str,
                        port_out_img name: str, port_out_ang: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    INPUT_G1_POS = 0
    # noinspection PyPep8Naming
    INPUT_G2_POS = 1
    # noinspection PyPep8Naming
    INPUT_G3_POS = 2
    # noinspection PyPep8Naming
    INPUT_G4_POS = 3
    # noinspection PyPep8Naming
    INPUT_G5_POS = 4
    # noinspection PyPep8Naming
    INPUT_G6_POS = 5
    # noinspection PyPep8Naming
    INPUT_G7_POS = 6
    # noinspection PyPep8Naming
    INPUT_G8_POS = 7
    # noinspection PyPep8Naming
    INPUT_G9_POS = 8
    # noinspection PyPep8Naming
    INPUT_WAVE_POS = 9
    # noinspection PyPep8Naming
    PORT_OUT_IMG_EDGE = 10
    # noinspection PyPep8Naming
    PORT_OUT_IMG_LINE = 11

    if len(param_list) != 12:
        log_error_to_console("FREI-CHEN JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        port_in_g1 = get_port_from_wave(name=param_list[INPUT_G1_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g2 = get_port_from_wave(name=param_list[INPUT_G2_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g3 = get_port_from_wave(name=param_list[INPUT_G3_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g4 = get_port_from_wave(name=param_list[INPUT_G4_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g5 = get_port_from_wave(name=param_list[INPUT_G5_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g6 = get_port_from_wave(name=param_list[INPUT_G6_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g7 = get_port_from_wave(name=param_list[INPUT_G7_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g8 = get_port_from_wave(name=param_list[INPUT_G8_POS], wave_offset=param_list[INPUT_WAVE_POS])
        port_in_g9 = get_port_from_wave(name=param_list[INPUT_G9_POS], wave_offset=param_list[INPUT_WAVE_POS])

        port_out_img_edge = get_port_from_wave(name=param_list[PORT_OUT_IMG_EDGE])
        port_out_img_line = get_port_from_wave(name=param_list[PORT_OUT_IMG_LINE])

        if (port_in_g1.is_valid() and port_in_g2.is_valid() and port_in_g3.is_valid() and port_in_g4.is_valid() and port_in_g5.is_valid()
                and port_in_g6.is_valid() and port_in_g7.is_valid() and port_in_g8.is_valid() and port_in_g9.is_valid()) is True:
            try:
                edge_space = np.zeros(shape=port_in_g1.arr.shape, dtype=np.float32)
                line_space = np.zeros(shape=port_in_g1.arr.shape, dtype=np.float32)
                space = np.zeros(shape=port_in_g1.arr.shape, dtype=np.float32)

                edge_space = edge_space + port_in_g1.arr + port_in_g2.arr + port_in_g3.arr + port_in_g4.arr
                line_space = line_space + port_in_g5.arr + port_in_g6.arr + port_in_g7.arr + port_in_g8.arr
                space = space + edge_space + line_space + port_in_g9.arr

                result_edge = np.sqrt(edge_space / space)
                result_lines = np.sqrt(line_space / space)

                port_out_img_edge.arr[:] = cv2.normalize(src=result_edge, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                         dtype=cv2.CV_8UC1)
                port_out_img_line.arr[:] = cv2.normalize(src=result_lines, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                         dtype=cv2.CV_8UC1)

                port_out_img_edge.set_valid()
                port_out_img_line.set_valid()
            except BaseException as error:
                log_error_to_console("FREI-CHEN JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


if __name__ == "__main__":
    pass
