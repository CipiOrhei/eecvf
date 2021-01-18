# noinspection PyPackageRequirements
import cv2
import numpy as np
# Do not delete used indirectly
# noinspection PyUnresolvedReferences
from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.port import Port

from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console

"""
Module handles the Morphological image processing jobs for the APPL block.
"""


def init_func() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def main_func_erosion(param_list: list = None) -> bool:
    """
    Erosion (usually represented by ⊖) is one of two fundamental operations (the other being dilation) in morphological image processing
    from which all other morphological operations are based.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1,  wave_offset, structural_element_cv_name, kernel_cv_size, number_iterations, custom_kernel,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_POS = 2
    # noinspection PyPep8Naming
    PORT_ITERATION_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 4

    if len(param_list) != 5:
        log_error_to_console("IMAGE EROSION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        port_out = get_port_from_wave(name=param_list[PORT_OUT_IMG])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                kernel = np.array(eval(param_list[PORT_KERNEL_POS]), dtype=np.uint8)
                port_out.arr[:] = cv2.erode(src=port_in.arr, kernel=kernel, iterations=param_list[PORT_ITERATION_POS])
                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("IMAGE EROSION JOB NOK: ", str(error))
                pass
        else:
            return False
        return True


def main_func_dilation(param_list: list = None) -> bool:
    """
    Dilation (usually represented by ⊕) is one of the basic operations in mathematical morphology. Originally developed for binary
    images, it has been expanded first to grayscale images, and then to complete lattices.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset, structural_element_cv_name, kernel_cv_size, number_iterations, custom_kernel,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_KERNEL_POS = 2
    # noinspection PyPep8Naming
    PORT_ITERATION_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 4

    if len(param_list) != 5:
        log_error_to_console("IMAGE DILATION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        port_out = get_port_from_wave(name=param_list[PORT_OUT_IMG])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                kernel = np.array(eval(param_list[PORT_KERNEL_POS]), dtype=np.uint8)
                port_out.arr[:] = cv2.dilate(src=port_in.arr, kernel=kernel, iterations=param_list[PORT_ITERATION_POS])
                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("IMAGE DILATION JOB NOK: ", str(error))
                pass
        else:
            return False
        return True


def main_func_morphology_ex(param_list: list = None) -> bool:
    """
    Morphological image processing is a collection of non-linear operations related to the shape or morphology of features in an image.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, structural_element_cv_name, kernel_cv_size, number_iterations,
                        custom_kernel,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_IN_OPERATION = 2
    # noinspection PyPep8Naming
    PORT_STRUCTURAL_ELEMENT_POS = 3
    # noinspection PyPep8Naming
    PORT_STRUCTURAL_KERNEL_SIZE_POS = 4
    # noinspection PyPep8Naming
    PORT_ITERATION_POS = 5
    # noinspection PyPep8Naming
    PORT_KERNEL_POS = 6
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 7

    if len(param_list) != 8:
        log_error_to_console("IMAGE CV2 MORPHOLOGICAL JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        port_out = get_port_from_wave(name=param_list[PORT_OUT_IMG])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                kernel_size = (param_list[PORT_STRUCTURAL_KERNEL_SIZE_POS], param_list[PORT_STRUCTURAL_KERNEL_SIZE_POS])

                if param_list[PORT_KERNEL_POS] == 'None':
                    kernel = cv2.getStructuringElement(shape=eval(param_list[PORT_STRUCTURAL_ELEMENT_POS]),
                                                       ksize=kernel_size)
                else:
                    kernel = np.array(eval(param_list[PORT_KERNEL_POS]), dtype=np.uint8)

                port_out.arr[:] = cv2.morphologyEx(src=port_in.arr, op=eval(param_list[PORT_IN_OPERATION]),
                                                   kernel=kernel, iterations=param_list[PORT_ITERATION_POS])
                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("IMAGE CV2 MORPHOLOGICAL JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def hit_and_miss(port_in: Port.arr, h: int, w: int, kernel_b1: np.ndarray, kernel_b2: np.ndarray):
    """
    Hit and miss algorithm.
    :param port_in: input image
    :param h: height of image
    :param w: width of image
    :param kernel_b1: first kernel
    :param kernel_b2: second kernel
    :return: True if the job executed OK.
    """
    img_255 = np.ones(shape=(h, w), dtype=np.uint8) * 255
    tmp = np.zeros(shape=(h, w), dtype=np.uint8)
    complement_img = img_255 - port_in

    for rotation in range(1, 5, 1):
        img_erode_b1 = cv2.erode(src=port_in, kernel=kernel_b1, iterations=1)
        img_erode_b2 = cv2.erode(src=complement_img, kernel=kernel_b2, iterations=1)
        tmp |= ((img_erode_b1 == 255) & (img_erode_b2 == 255))
        kernel_b1 = np.rot90(kernel_b1)
        kernel_b2 = np.rot90(kernel_b2)

    tmp = tmp * 255

    return cv2.normalize(src=tmp, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)


def thinning(input_image: Port.arr, h: int, w: int, kernel_b1: np.ndarray, kernel_b2: np.ndarray,
             kernel_b3: np.ndarray, kernel_b4: np.ndarray):
    """
    Thinning algorithm.
    :param input_image: input image
    :param h: height of image
    :param w: width of image
    :param kernel_b1: first kernel
    :param kernel_b2: second kernel
    :param kernel_b3: third kernel
    :param kernel_b4: forth kernel
    :return: True if the job executed OK.
    """
    img_255 = np.ones(shape=(h, w), dtype=np.uint8) * 255

    for rotation_axis in range(1, 5, 1):
        # do first hit-and-miss
        tmp = hit_and_miss(port_in=input_image, h=h, w=w, kernel_b1=kernel_b1, kernel_b2=kernel_b2)
        tmp = img_255 - tmp
        input_image = ((input_image == 255) & (tmp == 255)) * 255
        input_image = cv2.normalize(src=input_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # do second hit-and-miss
        tmp = hit_and_miss(port_in=input_image, h=h, w=w, kernel_b1=kernel_b3, kernel_b2=kernel_b4)
        tmp = img_255 - tmp
        input_image = ((input_image == 255) & (tmp == 255)) * 255
        input_image = cv2.normalize(src=input_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        kernel_b1 = np.rot90(kernel_b1)
        kernel_b2 = np.rot90(kernel_b2)
        kernel_b3 = np.rot90(kernel_b3)
        kernel_b4 = np.rot90(kernel_b4)

    return input_image


def main_func_hit_miss(param_list: list = None) -> bool:
    """
    The Hit-or-Miss transformation is useful to find patterns in binary images. In particular,
    it finds those pixels whose neighbourhood matches the shape of a first structuring element B1 while
    not matching the shape of a second structuring element B2 at the same time.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset, structural_element_cv_name, kernel_cv_size, number_iterations, custom_kernel,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_STRUCTURAL_ELEMENT_1_POS = 2
    # noinspection PyPep8Naming
    PORT_STRUCTURAL_ELEMENT_2_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 4

    if len(param_list) != 5:
        log_error_to_console("IMAGE CV2 MORPHOLOGICAL HIT MISS JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        port_out = get_port_from_wave(name=param_list[PORT_OUT_IMG])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                # noinspection PyPep8Naming
                kernel_B1 = np.array(eval(param_list[PORT_STRUCTURAL_ELEMENT_1_POS]), dtype=np.uint8)
                # noinspection PyPep8Naming
                kernel_B2 = np.array(eval(param_list[PORT_STRUCTURAL_ELEMENT_2_POS]), dtype=np.uint8)
                level = port_in.name.split('_')[-1]
                w = eval('global_var_handler.WIDTH_' + level)
                h = eval('global_var_handler.HEIGHT_' + level)

                port_out.arr[:] = hit_and_miss(port_in=port_in.arr.copy(), h=h, w=w, kernel_b1=kernel_B1, kernel_b2=kernel_B2)

                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("MORPHOLOGICAL HIT MISS JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_func_thinning(param_list: list = None) -> bool:
    """
    The Hit-or-Miss transformation is useful to find patterns in binary images. In particular,
    it finds those pixels whose neighbourhood matches the shape of a first structuring element B1 while
    not matching the shape of a second structuring element B2 at the same time.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image_1, wave_offset, structural_element_cv_name, kernel_cv_size, number_iterations,
                        custom_kernel,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_STRUCTURAL_ELEMENT_1_POS = 2
    # noinspection PyPep8Naming
    PORT_STRUCTURAL_ELEMENT_2_POS = 3
    # noinspection PyPep8Naming
    PORT_STRUCTURAL_ELEMENT_3_POS = 4
    # noinspection PyPep8Naming
    PORT_STRUCTURAL_ELEMENT_4_POS = 5
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 6

    if len(param_list) != 7:
        log_error_to_console("IMAGE CV2 MORPHOLOGICAL THINNING JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        port_out = get_port_from_wave(name=param_list[PORT_OUT_IMG])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                kernel_b1 = np.array(eval(param_list[PORT_STRUCTURAL_ELEMENT_1_POS]), dtype=np.uint8)
                kernel_b2 = np.array(eval(param_list[PORT_STRUCTURAL_ELEMENT_2_POS]), dtype=np.uint8)
                kernel_b3 = np.array(eval(param_list[PORT_STRUCTURAL_ELEMENT_3_POS]), dtype=np.uint8)
                kernel_b4 = np.array(eval(param_list[PORT_STRUCTURAL_ELEMENT_4_POS]), dtype=np.uint8)

                level = port_in.name.split('_')[-1]
                w = eval('global_var_handler.WIDTH_' + level)
                h = eval('global_var_handler.HEIGHT_' + level)

                port_out.arr[:] = thinning(input_image=port_in.arr, h=h, w=w, kernel_b1=kernel_b1, kernel_b2=kernel_b2,
                                           kernel_b3=kernel_b3, kernel_b4=kernel_b4)

                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("MORPHOLOGICAL THINNING JOB NOK: ", str(error))
                pass

        else:
            return False

        return True


if __name__ == "__main__":
    pass
