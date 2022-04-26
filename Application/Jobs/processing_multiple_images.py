import numpy as np
# noinspection PyPackageRequirements
import cv2
# Do not delete used indirectly
# noinspection PyUnresolvedReferences
from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console

# TODO change jobs to accept a list of images

"""
Module handles multiple image manipulation jobs for the APPL block.
"""


def init_func() -> JobInitStateReturn:
    """
    Init function for the job.
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def difference_2_matrix(param_list: list = None) -> bool:
    """
    Calculates the matrix that results by subtraction of a matrix from another
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset, image_2, wave_offset
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_1_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_1 = 1
    # noinspection PyPep8Naming
    PORT_IN_2_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE_2 = 3
    # noinspection PyPep8Naming
    PORT_OUT_VAL_POS = 4
    # noinspection PyPep8Naming
    PORT_IN_NORMALIZE = 5
    # noinspection PyPep8Naming
    PORT_IN_CMAP = 6

    if len(param_list) != 7:
        log_error_to_console("MATRIX DIFFERENCE CALCULATION INTENSITY JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1 = get_port_from_wave(name=param_list[PORT_IN_1_POS], wave_offset=param_list[PORT_IN_WAVE_1])
        p_in_2 = get_port_from_wave(name=param_list[PORT_IN_2_POS], wave_offset=param_list[PORT_IN_WAVE_2])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_VAL_POS])

        if param_list[PORT_IN_CMAP] is True:
            p_out_cmap = get_port_from_wave(name='CMAP_' + param_list[PORT_OUT_VAL_POS])

        if p_in_1.is_valid() is True and p_in_2.is_valid() is True:
            if p_in_1.arr.shape == p_in_2.arr.shape:
                # try:
                if True:
                    img = cv2.subtract(src1=p_in_1.arr.astype('int32'), src2=p_in_2.arr.astype('int32'))

                    if param_list[PORT_IN_NORMALIZE] is True:
                        img = abs(img)

                        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                        # ret, img = cv2.threshold(src=img, thresh=2, maxval=255, type=cv2.THRESH_BINARY)

                    if param_list[PORT_IN_CMAP] is True:
                        import matplotlib.pyplot as plt
                        # import color_utils
                        # a colormap and a normalization instance
                        cmap = plt.cm.cividis
                        # cmap = plt.cm.bone
                        norm = plt.Normalize(vmin=img.min(), vmax=img.max())

                        # map the normalized data to colors
                        # image is now RGBA (512x512x4)
                        cmap_img = cmap(norm(img)).astype('float32')
                        cmap_img = cv2.cvtColor(cmap_img, cv2.COLOR_RGBA2BGR)
                        cmap_img = cv2.normalize(src=cmap_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

                        p_out_cmap.arr[:] = cmap_img[:]
                        p_out_cmap.set_valid()

                        # cv2.imshow('t', p_out_cmap.arr)
                        # cv2.waitKey(0)

                    p_out.arr = img
                    p_out.set_valid()
                # except BaseException as error:
                #     log_error_to_console("MATRIX DIFFERENCE JOB NOK: ", str(error))
                #     pass
            else:
                log_error_to_console("MATRIX DIFFERENCE INPUTS NOT SAME SHAPE", str(len(param_list)))
                return False
        else:
            return False

        return True


def difference_2_matrix_1_px_offset(param_list: list = None) -> bool:
    """
    Calculates the matrix that results by subtraction of a matrix from another accounting for 1 px offset in each direction
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset_1, image_2, wave_offset_2
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_1_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_1 = 1
    # noinspection PyPep8Naming
    PORT_IN_2_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE_2 = 3
    # noinspection PyPep8Naming
    PORT_OUT_VAL_POS = 4

    if len(param_list) != 5:
        log_error_to_console("MATRIX DIFFERENCE 1 PX OFFSET CALCULATION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1 = get_port_from_wave(name=param_list[PORT_IN_1_POS], wave_offset=param_list[PORT_IN_WAVE_1])
        p_in_2 = get_port_from_wave(name=param_list[PORT_IN_2_POS], wave_offset=param_list[PORT_IN_WAVE_2])
        p_out_1 = get_port_from_wave(name=param_list[PORT_OUT_VAL_POS])

        if p_in_1.is_valid() is True and p_in_2.is_valid() is True:
            if p_in_1.arr.shape == p_in_2.arr.shape:
                try:
                    one_line_zeroes = np.zeros(shape=(1, p_in_1.arr.shape[1]), dtype=np.uint8)
                    one_column_zeroes = np.zeros(shape=(p_in_1.arr.shape[0], 1), dtype=np.uint8)
                    one_column_zeroes_1 = np.zeros(shape=(p_in_1.arr.shape[0] - 1, 1), dtype=np.uint8)

                    center = cv2.subtract(src1=p_in_1.arr, src2=p_in_2.arr)
                    ret, center = cv2.threshold(src=center, thresh=2, maxval=255, type=cv2.THRESH_BINARY)

                    one_left = cv2.subtract(src1=p_in_1.arr, src2=np.concatenate((p_in_2.arr[:, 1:], one_column_zeroes), axis=1))
                    ret, one_left = cv2.threshold(src=one_left, thresh=2, maxval=255, type=cv2.THRESH_BINARY)

                    one_right = cv2.subtract(src1=p_in_1.arr, src2=np.concatenate((one_column_zeroes, p_in_2.arr[:, :-1]), axis=1))
                    ret, one_right = cv2.threshold(src=one_right, thresh=2, maxval=255, type=cv2.THRESH_BINARY)

                    one_up = cv2.subtract(src1=p_in_1.arr, src2=np.concatenate((p_in_2.arr[1:, :], one_line_zeroes), axis=0))
                    ret, one_up = cv2.threshold(src=one_up, thresh=2, maxval=255, type=cv2.THRESH_BINARY)

                    one_down = cv2.subtract(src1=p_in_1.arr, src2=np.concatenate((one_line_zeroes, p_in_2.arr[:-1, :]), axis=0))
                    ret, one_down = cv2.threshold(src=one_down, thresh=2, maxval=255, type=cv2.THRESH_BINARY)

                    one_left_up = cv2.subtract(src1=p_in_1.arr,
                                               src2=np.concatenate(
                                                   (np.concatenate((p_in_2.arr[1:, 1:], one_column_zeroes_1), axis=1), one_line_zeroes),
                                                   axis=0))
                    ret, one_left_up = cv2.threshold(src=one_left_up, thresh=2, maxval=255, type=cv2.THRESH_BINARY)

                    one_left_down = cv2.subtract(src1=p_in_1.arr,
                                                 src2=np.concatenate(
                                                     (one_line_zeroes, np.concatenate((p_in_2.arr[:-1, 1:], one_column_zeroes_1), axis=1)),
                                                     axis=0))
                    ret, one_left_down = cv2.threshold(src=one_left_down, thresh=2, maxval=255, type=cv2.THRESH_BINARY)

                    one_right_down = cv2.subtract(src1=p_in_1.arr,
                                                  src2=np.concatenate((one_line_zeroes,
                                                                       np.concatenate((one_column_zeroes_1, p_in_2.arr[:-1, :-1]), axis=1)),
                                                                      axis=0))
                    ret, one_right_down = cv2.threshold(src=one_right_down, thresh=2, maxval=255, type=cv2.THRESH_BINARY)

                    one_right_up = cv2.subtract(src1=p_in_1.arr,
                                                src2=np.concatenate(
                                                    (np.concatenate((one_column_zeroes_1, p_in_2.arr[1:, :-1]), axis=1), one_line_zeroes),
                                                    axis=0))
                    ret, one_right_up = cv2.threshold(src=one_right_up, thresh=2, maxval=255, type=cv2.THRESH_BINARY)

                    img = ((center != 0) & (one_left != 0) & (one_right != 0) & (one_up != 0) & (one_down != 0) &
                           (one_left_up != 0) & (one_left_down != 0) & (one_right_down != 0) & (one_right_up != 0)) * 255

                    p_out_1.arr[:] = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                    p_out_1.set_valid()
                except BaseException as error:
                    log_error_to_console("MATRIX DIFFERENCE 1 PX OFFSET JOB NOK: ", str(error))
                    pass
            else:
                log_error_to_console("MATRIX DIFFERENCE 1 PX OFFSET INPUTS NOT SAME SHAPE", str(len(param_list)))
                return False
        else:
            return False

        return True


def add_2_matrix(param_list: list = None) -> bool:
    """
    Calculates the matrix that results by the sum of 2 matrices.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset_1, image_2, wave_offset_2
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_1_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_1 = 1
    # noinspection PyPep8Naming
    PORT_IN_2_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE_2 = 3
    # noinspection PyPep8Naming
    PORT_OUT_VAL_POS = 4
    # noinspection PyPep8Naming
    PORT_NORMALIZE = 5

    if len(param_list) != 6:
        log_error_to_console("MATRIX SUM CALCULATION INTENSITY JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1 = get_port_from_wave(name=param_list[PORT_IN_1_POS], wave_offset=param_list[PORT_IN_WAVE_1])
        p_in_2 = get_port_from_wave(name=param_list[PORT_IN_2_POS], wave_offset=param_list[PORT_IN_WAVE_2])
        p_out_1 = get_port_from_wave(name=param_list[PORT_OUT_VAL_POS])

        if p_in_1.is_valid() is True and p_in_2.is_valid() is True:
            if p_in_1.arr.shape == p_in_2.arr.shape:
                try:
                    result = cv2.add(src1=p_in_1.arr, src2=p_in_2.arr)
                    if param_list[PORT_NORMALIZE] is False:
                        ret, p_out_1.arr[:] = cv2.threshold(src=result, thresh=255, maxval=255, type=cv2.THRESH_TRUNC)
                    else:
                        p_out_1.arr[:] = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                       dtype=cv2.CV_8UC1)
                    p_out_1.set_valid()
                except BaseException as error:
                    log_error_to_console("MATRIX SUM CALCULATION JOB NOK: ", str(error))
                    pass
            else:
                log_error_to_console("MATRIX SUM INPUTS NOT SAME SHAPE", str(len(param_list)))
                return False
        else:
            return False

        return True


def and_bitwise_between_2_images(param_list: list = None) -> bool:
    """
    Calculates the matrix that results by applying bitwise AND for the 2 images.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset_1, image_2, wave_offset_2,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_1_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_1 = 1
    # noinspection PyPep8Naming
    PORT_IN_2_IMG_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE_2 = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 4

    if len(param_list) != 5:
        log_error_to_console("IMAGE BITWISE AND JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1_image = get_port_from_wave(name=param_list[PORT_IN_1_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_1])
        p_in_2_image = get_port_from_wave(name=param_list[PORT_IN_2_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_2])
        p_out_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_1_image.is_valid() is True and p_in_2_image.is_valid() is True:
            try:
                p_out_out.arr[:] = ((p_in_1_image.arr != 0) & (p_in_2_image.arr != 0)) * 255
                p_out_out.set_valid()
            except BaseException as error:
                log_error_to_console("IMAGE BITWISE AND JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def or_bitwise_between_2_images(param_list: list = None) -> bool:
    """
    Calculates the matrix that results by applying bitwise OR for the 2 images
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset_1, image_2, wave_offset_2,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_1_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_1 = 1
    # noinspection PyPep8Naming
    PORT_IN_2_IMG_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE_2 = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 4

    if len(param_list) != 5:
        log_error_to_console("IMAGE BITWISE OR JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1_image = get_port_from_wave(name=param_list[PORT_IN_1_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_1])
        p_in_2_image = get_port_from_wave(name=param_list[PORT_IN_2_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_2])
        p_out_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_1_image.is_valid() is True and p_in_2_image.is_valid() is True:
            try:
                p_out_out.arr[:] = ((p_in_1_image.arr != 255) | (p_in_2_image.arr != 255)) * 255
                p_out_out.set_valid()
            except BaseException as error:
                log_error_to_console("BITWISE OR JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def or_bitwise_between_4_images(param_list: list = None) -> bool:
    """
    Calculates the matrix that results by applying bitwise OR for the 4 images
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset_1, image_2, wave_offset_2, image_3, wave_offset_3, image_4, wave_offset_4,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_1_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_1 = 1
    # noinspection PyPep8Naming
    PORT_IN_2_IMG_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE_2 = 3
    # noinspection PyPep8Naming
    PORT_IN_3_IMG_POS = 4
    # noinspection PyPep8Naming
    PORT_IN_WAVE_3 = 5
    # noinspection PyPep8Naming
    PORT_IN_4_IMG_POS = 6
    # noinspection PyPep8Naming
    PORT_IN_WAVE_4 = 7
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 8

    if len(param_list) != 9:
        log_error_to_console("FOUR IMAGE BITWISE OR JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1_image = get_port_from_wave(name=param_list[PORT_IN_1_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_1])
        p_in_2_image = get_port_from_wave(name=param_list[PORT_IN_2_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_2])
        p_in_3_image = get_port_from_wave(name=param_list[PORT_IN_3_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_3])
        p_in_4_image = get_port_from_wave(name=param_list[PORT_IN_4_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_4])
        p_out_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_1_image.is_valid() is True and p_in_2_image.is_valid() is True \
                and p_in_3_image.is_valid() is True and p_in_4_image.is_valid() is True:
            try:
                p_out_out.arr[:] = ((p_in_1_image.arr != 255) | (p_in_2_image.arr != 255) |
                                    (p_in_3_image.arr != 255) | (p_in_4_image.arr != 255)) * 255
                p_out_out.set_valid()
            except BaseException as error:
                log_error_to_console("FOUR IMAGE BITWISE OR JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def xor_bitwise_between_2_images(param_list: list = None) -> bool:
    """
    Calculates the matrix that results by applying bitwise XOR for the 2 images
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset_1, image_2, wave_offset_2,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_1_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_1 = 1
    # noinspection PyPep8Naming
    PORT_IN_2_IMG_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE_2 = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 4

    if len(param_list) != 5:
        log_error_to_console("IMAGE BITWISE XOR JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1_image = get_port_from_wave(name=param_list[PORT_IN_1_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_1])
        p_in_2_image = get_port_from_wave(name=param_list[PORT_IN_2_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_2])
        p_out_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_1_image.is_valid() is True and p_in_2_image.is_valid() is True:
            try:
                p_out_out.arr[:] = ((p_in_1_image.arr != 255) ^ (p_in_2_image.arr != 255)) * 255
                p_out_out.set_valid()
            except BaseException as error:
                log_error_to_console("BITWISE XOR JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def intersect_between_2_images(param_list: list = None) -> bool:
    """
    Calculates the matrix that results intersections of 2 matrix.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset_1, image_2, wave_offset_2,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_1_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_1 = 1
    # noinspection PyPep8Naming
    PORT_IN_MASK_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE_2 = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 4

    if len(param_list) != 5:
        log_error_to_console("IMAGE INTERSECT JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1_image = get_port_from_wave(name=param_list[PORT_IN_1_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_1])
        p_in_2_image = get_port_from_wave(name=param_list[PORT_IN_MASK_POS], wave_offset=param_list[PORT_IN_WAVE_2])
        p_out_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_1_image.is_valid() is True and p_in_2_image.is_valid() is True:
            try:
                p_out_out.arr[:] = ((p_in_1_image.arr != 0) & (p_in_2_image.arr != 0)) * p_in_1_image.arr
                p_out_out.set_valid()
            except BaseException as error:
                log_error_to_console("IMAGE INTERSECT JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


if __name__ == "__main__":
    pass
