import numpy as np
# noinspection PyPackageRequirements
import cv2
import skimage
from scipy import ndimage
from skimage import exposure
from PIL import Image
# Do not delete used indirectly
# noinspection PyUnresolvedReferences
from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_file, log_error_to_console

"""
Module handles single image jobs for the APPL block.
"""


def init_func_global(param_list: list = None) -> JobInitStateReturn:
    """
    Init function for the job.
    :param param_list: list of PORT to be written in the csv file
    :return: INIT or NOT_INIT state for the job
    """
    if param_list is not None:
        log_to_file(param_list[0] + ' ')

    return JobInitStateReturn(True)


def init_func() -> JobInitStateReturn:
    """
    Init function for the job.
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)


def median_calculation(param_list: list = None) -> bool:
    """
    Calculates the median pixel value.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, wave_offset
                        port_out: value]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_OUT_VAL_POS = 2

    if len(param_list) != 3:
        log_error_to_console("MEDIAN CALCULATION INTENSITY JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1 = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        p_out_1 = get_port_from_wave(name=param_list[PORT_OUT_VAL_POS])

        if p_in_1.is_valid() is True:
            try:
                p_out_1.arr[:] = int(np.median(p_in_1.arr))
                log_to_file(p_out_1.arr[0].__str__())
                p_out_1.set_valid()
            except BaseException as error:
                log_error_to_console("MEDIAN PIXEL CALCULATION JOB NOK: ", str(error))
                log_to_file('')
                pass
        else:
            log_to_file('')
            return False

        return True


def nr_edge_px_calculation(param_list: list = None) -> bool:
    """
    Calculates the number of edge pixels
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, port_in_wave: wave for input image, port_out: value]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_OUT_VAL_POS = 2

    if len(param_list) != 3:
        log_error_to_console("NR EDGE PX CALCULATION JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1 = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        p_out_1 = get_port_from_wave(name=param_list[PORT_OUT_VAL_POS])

        if p_in_1.is_valid() is True:
            try:
                p_out_1.arr[:] = np.count_nonzero(a=p_in_1.arr)
                log_to_file(p_out_1.arr[0].__str__())
                p_out_1.set_valid()
            except BaseException as error:
                log_to_file('')
                log_error_to_console("NR EDGE PX CALCULATION JOB NOK: ", str(error))
                pass

        else:
            log_to_file('')
            return False

        return True


def mean_calculation(param_list: list = None) -> bool:
    """
    Calculates the mean pixel value
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, wave
                        port_out: value]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_OUT_VAL_POS = 2

    if len(param_list) != 3:
        log_error_to_console("MEAN CALCULATION INTENSITY JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1 = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        p_out_1 = get_port_from_wave(name=param_list[PORT_OUT_VAL_POS])

        if p_in_1.is_valid() is True:
            try:
                p_out_1.arr[:] = int(np.mean(p_in_1.arr))
                log_to_file(p_out_1.arr[0].__str__())
                p_out_1.set_valid()
            except BaseException as error:
                log_to_file('')
                log_error_to_console("MEAN CALCULATION INTENSITY JOB NOK: ", str(error))
                pass
        else:
            log_to_file('')
            return False

        return True


def max_calculation(param_list: list = None) -> bool:
    """
    Calculates the maximum pixel value of an greyscale image.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, wave_offset
                        port_out: value]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_OUT_VAL_POS = 2

    if len(param_list) != 3:
        log_error_to_console("MAX CALCULATION INTENSITY JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_1 = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        p_out_1 = get_port_from_wave(name=param_list[PORT_OUT_VAL_POS])

        if p_in_1.is_valid() is True:
            try:
                p_out_1.arr[:] = int(np.max(p_in_1.arr))
                log_to_file(p_out_1.arr[0].__str__())
                p_out_1.set_valid()
            except BaseException as error:
                log_error_to_console("MAX PIXEL CALCULATION JOB NOK: ", str(error))
                log_to_file('')
                pass
        else:
            log_to_file('')
            return False

        return True


def add_gaussian_noise(param_list: list = None) -> bool:
    """
    Calculates gaussian noise over an image
    :param param_list: Param needed to respect the following list:
                       [port_in_name: image, port_in_wave: wave offset, mean_value, mean_variance,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_IN_MEAN_POS = 2
    # noinspection PyPep8Naming
    PORT_IN_VARIANCE_POS = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 4

    if len(param_list) != 5:
        log_error_to_console("GAUSSIAN NOISE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        p_out_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                if param_list[PORT_IN_MEAN_POS] != 0.0 or param_list[PORT_IN_VARIANCE_POS] != 0.0:
                    result = skimage.util.random_noise(image=p_in_image.arr, mode="gaussian",
                                                       mean=param_list[PORT_IN_MEAN_POS], var=param_list[PORT_IN_VARIANCE_POS])
                    p_out_out.arr[:] = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                else:
                    p_out_out.arr[:] = p_in_image.arr.copy()

                p_out_out.set_valid()
            except BaseException as error:
                log_error_to_console("GAUSSIAN NOISE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def add_salt_pepper_noise(param_list: list = None) -> bool:
    """
    Calculates salt and pepper noise over an image.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, wave_offset, density_value,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_IN_DENSITY_POS = 2
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 3

    if len(param_list) != 4:
        log_error_to_console("SALT&PEPPER NOISE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                if param_list[PORT_IN_DENSITY_POS] != 0:
                    result = skimage.util.random_noise(image=p_in_image.arr.copy(), mode="s&p", salt_vs_pepper=param_list[PORT_IN_DENSITY_POS])
                    p_out.arr[:] = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                else:
                    p_out.arr[:] = p_in_image.arr.copy()
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("SALT&PEPPER NOISE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def add_speckle_noise(param_list: list = None) -> bool:
    """
    Calculates speckle noise over an image.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, wave_offset, variance_value,
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_IN_VARIANCE_POS = 2
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 3

    if len(param_list) != 4:
        log_error_to_console("SPECKLE NOISE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        p_out_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                result = skimage.util.random_noise(image=p_in_image.arr, mode="speckle", var=param_list[PORT_IN_VARIANCE_POS])
                p_out_out.arr[:] = cv2.normalize(src=result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                p_out_out.set_valid()
            except BaseException as error:
                log_error_to_console("SPECKLE NOISE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def transform_to_greyscale(param_list: list = None) -> bool:
    """
    Transforming RGB image to grayscale image.
    :param param_list: Param needed to respect the following list:
                       [port_in_name: image, port_in_wave: wave offset, port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_IMG_WAVE = 1
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 2

    if len(param_list) != 3:
        log_error_to_console("GRAYSCALE TRANSFORM JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_IMG_WAVE])
        p_out_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                p_out_out.arr[:] = cv2.cvtColor(src=p_in_image.arr, code=cv2.COLOR_BGR2GRAY)
                p_out_out.set_valid()
            except BaseException as error:
                log_error_to_console("GRAYSCALE TRANSFORM JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def image_complement(param_list: list = None) -> bool:
    """
    Calculates the matrix that results by subtraction of a matrix from another
    :param param_list: Param needed to respect the following list:
                       [port_in_name: image, port_in_wave: wave offset, port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 2

    if len(param_list) != 3:
        log_error_to_console("IMAGE COMPLEMENT JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        p_out_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                p_out_out.arr[:] = cv2.bitwise_not(src=p_in_image.arr)
                p_out_out.set_valid()
            except BaseException as error:
                log_error_to_console("IMAGE COMPLEMENT JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def do_image_crop(param_list: list = None) -> bool:
    """
    Do image cropping function.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, port_in_wave: wave of image,
                        start_width : proportion of image width where to start, end_width : proportion of image width where to end
                        start_height : proportion of image height where to start, end_height : proportion of image height where to end
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_START_WIDTH = 2
    # noinspection PyPep8Naming
    PORT_IN_END_WIDTH = 3
    # noinspection PyPep8Naming
    PORT_IN_START_HEIGHT = 4
    # noinspection PyPep8Naming
    PORT_IN_END_HEIGHT = 5
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 6

    if len(param_list) != 7:
        log_error_to_console("IMAGE CROP JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                s_w = int(param_list[PORT_IN_START_WIDTH] / 100 * p_in_image.arr.shape[1])
                e_w = int(param_list[PORT_IN_END_WIDTH] / 100 * p_in_image.arr.shape[1])
                s_h = int(param_list[PORT_IN_START_HEIGHT] / 100 * p_in_image.arr.shape[0])
                e_h = int(param_list[PORT_IN_END_HEIGHT] / 100 * p_in_image.arr.shape[0])

                p_out.arr[s_h:e_h, s_w:e_w] = p_in_image.arr[s_h:e_h, s_w:e_w]

                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("IMAGE CROP JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def rotate_main(param_list: list = None) -> bool:
    """
    Do image cropping function.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, port_in_wave: wave of image, angle: to rotate, reshape: if we want to reshape, extend_border:
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_ANGLE = 2
    # noinspection PyPep8Naming
    PORT_IN_RESHAPE = 3
    # noinspection PyPep8Naming
    PORT_IN_BORDER = 4
    # noinspection PyPep8Naming
    PORT_INTERPOLATION = 5
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 6

    if len(param_list) != 7:
        log_error_to_console("ROTATE IMAGE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                mode = 'constant'
                order = 3
                prefilter = True

                if param_list[PORT_IN_BORDER] is True:
                    mode = 'reflect'
                if param_list[PORT_INTERPOLATION] is False:
                    order = 0
                    prefilter = False

                # TODO fix this so we can have a fixed port size from beginning
                p_out.arr = ndimage.rotate(input=p_in_image.arr.copy(), angle=param_list[PORT_IN_ANGLE],
                                              reshape=param_list[PORT_IN_RESHAPE], mode=mode, order=order, prefilter=prefilter)

                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("ROTATE IMAGE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def flip_main(param_list: list = None) -> bool:
    """
    Do image cropping function.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, port_in_wave: wave of image, angle: to rotate, reshape: if we want to reshape, extend_border:
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_FLIP_HORIZONTAL = 2
    # noinspection PyPep8Naming
    PORT_IN_FLIP_VERTICAL = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 4

    if len(param_list) != 5:
        log_error_to_console("FLIP IMAGE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                if param_list[PORT_IN_FLIP_HORIZONTAL] is True and param_list[PORT_IN_FLIP_VERTICAL] is True:
                    p_out.arr[:] = cv2.flip(src=p_in_image.arr.copy(), flipCode=-1)
                elif param_list[PORT_IN_FLIP_HORIZONTAL] is True and param_list[PORT_IN_FLIP_VERTICAL] is False:
                    p_out.arr[:] = cv2.flip(src=p_in_image.arr.copy(), flipCode=1)
                elif param_list[PORT_IN_FLIP_HORIZONTAL] is False and param_list[PORT_IN_FLIP_VERTICAL] is True:
                    p_out.arr[:] = cv2.flip(src=p_in_image.arr.copy(), flipCode=0)
                else:
                    p_out.arr[:] = p_in_image.arr.copy()

                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("FLIP IMAGE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def zoom_main(param_list: list = None) -> bool:
    """
    Do image cropping function.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, port_in_wave: wave of image, angle: to rotate, reshape: if we want to reshape, extend_border:
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_ZOOM = 2
    # noinspection PyPep8Naming
    PORT_IN_INTERPOLATION_ZOOM = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 4

    if len(param_list) != 5:
        log_error_to_console("ZOOM IMAGE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                h, w = p_in_image.arr.shape[:2]
                zoom_tuple = (param_list[PORT_IN_ZOOM],) * 2 + (1,) * (p_in_image.arr.ndim - 2)

                interpolation_order = 3
                interpolation_prefilter = True

                if param_list[PORT_IN_INTERPOLATION_ZOOM] is False:
                    interpolation_order = 0
                    interpolation_prefilter = False

                new_h = np.ceil(param_list[PORT_IN_ZOOM] * h - 0.5)
                new_w = np.ceil(param_list[PORT_IN_ZOOM] * w - 0.5)

                top_h = int(abs(new_h - h) // 2)
                top_w = int(abs(new_w - w) // 2)

                # Zooming out
                if param_list[PORT_IN_ZOOM] < 1.0:
                    # Bounding box of the zoomed-out image within the output array
                    # Zero-padding
                    out = np.zeros_like(p_in_image.arr)
                    out[top_h:h - top_h, top_w:w - top_w] = ndimage.zoom(input=p_in_image.arr.copy(),
                                                                         zoom=zoom_tuple,
                                                                         order=interpolation_order, prefilter=interpolation_prefilter)
                    p_out.arr[:] = out[:]
                # Zooming in
                elif param_list[PORT_IN_ZOOM] > 1.0:
                    out = ndimage.zoom(input=p_in_image.arr.copy(),
                                       zoom=zoom_tuple,
                                       order=interpolation_order, prefilter=interpolation_prefilter)

                    p_out.arr[:] = out[top_h:top_h + h, top_w:top_w + w]
                # If zoom_factor == 1, just return the input array
                else:
                    p_out.arr[:] = p_in_image.arr

                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("FLIP IMAGE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def change_contrast_brightness_main(param_list: list = None) -> bool:
    """
    Do image cropping function.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, port_in_wave: wave of image, angle: to rotate, reshape: if we want to reshape, extend_border:
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_ALPHA = 2
    # noinspection PyPep8Naming
    PORT_IN_BETA = 3
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 4

    if len(param_list) != 5:
        log_error_to_console("CHANGE CONTRAST BRIGHTNESS IMAGE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                if param_list[PORT_IN_ALPHA] != 1.0 or param_list[PORT_IN_BETA] != 0:
                    p_out.arr[:] = cv2.convertScaleAbs(src=p_in_image.arr.copy(), alpha=param_list[PORT_IN_ALPHA],
                                                       beta=param_list[PORT_IN_BETA])
                else:
                    p_out.arr[:] = p_in_image.arr.copy()

                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("ROTATE IMAGE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def gamma_correction_main(param_list: list = None) -> bool:
    """
    Gamma correction can be used to correct the brightness of an image by using a non linear transformation between the input values and
    the mapped output.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, port_in_wave: wave of image, angle: to rotate, reshape: if we want to reshape, extend_border:
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_GAMMA = 2
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 3

    if len(param_list) != 4:
        log_error_to_console("GAMMA CONTROL IMAGE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                p_out.arr[:] = exposure.adjust_gamma(image=p_in_image.arr.copy(), gamma=param_list[PORT_IN_GAMMA])

                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("GAMMA CONTROL IMAGE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def pixelate_main(param_list: list = None) -> bool:
    """
    Pixelate effect.
    :param param_list: Param needed to respect the following list:
                       [port_in name: image, port_in_wave: wave of image, angle: to rotate, reshape: if we want to reshape, extend_border:
                        port_out: image_result]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_IMG_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_PIXELATE_RATION = 2
    # noinspection PyPep8Naming
    PORT_OUT_IMG_POS = 3

    if len(param_list) != 4:
        log_error_to_console("PIXELATE IMAGE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        p_in_image = get_port_from_wave(name=param_list[PORT_IN_IMG_POS], wave_offset=param_list[PORT_IN_WAVE])
        p_out = get_port_from_wave(name=param_list[PORT_OUT_IMG_POS])

        if p_in_image.is_valid() is True:
            try:
                ratio_h = int(np.ceil(p_in_image.arr.shape[0]/param_list[PORT_IN_PIXELATE_RATION]))
                ratio_w = int(np.ceil(p_in_image.arr.shape[1]/param_list[PORT_IN_PIXELATE_RATION]))
                img = Image.fromarray(p_in_image.arr)
                # Resize smoothly down to 16x16 pixels
                imgSmall = img.resize((ratio_w, ratio_h), resample=Image.NEAREST)
                # Scale back up using NEAREST to original size
                result = imgSmall.resize(img.size, Image.NEAREST)
                p_out.arr[:] = np.array(result.getdata()).reshape(p_in_image.arr.shape)
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("PIXELATE IMAGE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


if __name__ == "__main__":
    pass
