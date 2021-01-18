# noinspection PyPackageRequirements
import cv2
import numpy as np
from numba import jit

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave, Port
from Utils.log_handler import log_error_to_console
# Do not delete used indirectly
# noinspection PyUnresolvedReferences
import Application.Jobs.kernels


def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """

    return JobInitStateReturn(True)


def init_func_isef() -> JobInitStateReturn:
    """
    Init function for the job
    :return: INIT or NOT_INIT state for the job
    """
    # fake call for ISEF_filter function to push python to pipeline
    isef_filter(np.ones((5, 5)), 0.5)

    return JobInitStateReturn(True)


@jit(nopython=True)
def apply_isef_vertical(image: Port.arr, b: float, b1: float, b2: float) -> Port.arr:
    """
    Apply the filter in the vertical direction (to the rows)
    :param image: image to apply on
    :param b: smoothing factor
    :param b1: smoothing factor discrete component
    :param b2: smoothing factor discrete component
    :return: resulting image
    """
    new_image_a = new_image_b = new_image = np.zeros(image.shape)

    # compute boundary conditions
    new_image_a[0, :] = b1 * image[0, :]
    new_image_b[-1, :] = b2 * image[-1, :]

    # compute causal component
    for row in range(1, image.shape[0], 1):
        for col in range(0, image.shape[1], 1):
            new_image_a[row][col] = b1 * image[row][col] + b * new_image_a[row - 1][col]

    # compute anti-causal component
    for row in range(image.shape[0] - 2, -1, -1):
        for col in range(0, image.shape[1], 1):
            new_image_b[row][col] = b2 * image[row][col] + b * new_image_b[row + 1][col]

    # boundary case for computing output of first filter
    new_image[-1, :] = new_image_a[-1, :]

    # now compute the output of the first filter this is the sum of the causal and anti-causal components
    for row in range(0, image.shape[0] - 2, 1):
        for col in range(0, image.shape[1], 1):
            new_image[row][col] = new_image_a[row][col] + new_image_b[row + 1][col]

    return new_image


@jit(nopython=True)
def apply_isef_horizontal(image: Port.arr, b: float, b1: float, b2: float) -> Port.arr:
    """
    Apply the filter in the horizontal direction (to the columns)
    :param image: image to apply on
    :param b: smoothing factor
    :param b1: smoothing factor discrete component
    :param b2: smoothing factor discrete component
    :return: resulting image
    """
    new_image_a = new_image_b = new_image = np.zeros(image.shape)

    # compute boundary conditions
    new_image_a[:, 0] = b1 * image[:, 0]
    new_image_b[:, -1] = b2 * image[:, -1]

    # compute causal component
    for col in range(1, image.shape[1], 1):
        for row in range(0, image.shape[0], 1):
            new_image_a[row][col] = b1 * image[row][col] + b * new_image_a[row][col - 1]

    # compute anti-causal component
    for col in range(image.shape[1] - 2, -1, -1):
        for row in range(0, image.shape[0], 1):
            new_image_b[row][col] = b2 * image[row][col] + b * new_image_b[row][col + 1]

    # boundary case for computing output of first filter
    new_image[:, -1] = new_image_a[:, -1]

    # now compute the output of the first filter this is the sum of the causal and anti-causal components
    for row in range(0, image.shape[0], 1):
        for col in range(0, image.shape[1] - 1, 1):
            new_image[row][col] = new_image_a[row][col] + new_image_b[row][col + 1]

    return new_image


def isef_filter(image: Port.arr, smoothing_factor: float = 0.9) -> Port.arr:
    """
    Calculate the infinite symmetric exponential filter (ISEF)
    :param image: image to apply on
    :param smoothing_factor: smoothing factor
    :return: resulting image
    """
    smoothing_factor_b1 = (1.0 - smoothing_factor) / (1.0 + smoothing_factor)
    smoothing_factor_b2 = smoothing_factor * smoothing_factor_b1

    isef_image = apply_isef_vertical(image=image, b=smoothing_factor, b1=smoothing_factor_b1, b2=smoothing_factor_b2)
    isef_image = apply_isef_horizontal(image=isef_image, b=smoothing_factor, b1=smoothing_factor_b1, b2=smoothing_factor_b2)

    return isef_image


@jit(nopython=True)
def is_candidate_edge(image, original_image, row, col):
    """
    Finds zero-crossings in laplacian (buff) orig is the smoothed image.
    Test for zero-crossings of laplacian then make sure that zero-crossing sign
    correspondence principle is satisfied. i.e. a positive z-c must have a positive 1st derivative
    where positive z-c means the 2nd deriv goes from positive to negative as we pass through the step edge
    :param image: laplacian image
    :param original_image: original image on which the laplace was performed(smoothed one)
    :param row: row of the image
    :param col: column of the image
    :return: if edge point is a candidate for zero crossing
    """
    if image[row][col] != 0 and image[row + 1][col] == 0:
        # positive z-c
        if original_image[row + 1][col] - original_image[row - 1][col] > 0:
            return 1
        else:
            return 0
    elif image[row][col] != 0 and image[row][col + 1] == 0:
        # positive z-c
        if original_image[row][col + 1] - original_image[row][col - 1] > 0:
            return 1
        else:
            return 0
    elif image[row][col] != 0 and image[row - 1][col] == 0:
        # negative z-c
        if original_image[row + 1][col] - original_image[row - 1][col] < 0:
            return 1
        else:
            return 0
    elif image[row][col] != 0 and image[row][col - 1] == 0:
        # negative z-c
        if original_image[row][col + 1] - original_image[row][col - 1] < 0:
            return 1
        else:
            return 0
    else:
        # not a z-c
        return 0


@jit(nopython=True)
def compute_adaptive_gradient(image, original_image, window_size, row, col):
    """
    The best estimate of the gradient at that point should be the difference in level between the two regions,
    where one region corresponds to the zero pixels in the BLI and the other corresponds to the one-valued pixels.
    :param image: laplacian image
    :param original_image: original image on which the laplace was performed(smoothed one)
    :param window_size: size of gradient window
    :param row: row of the image
    :param col: column of the image
    :return: value of pixel after adaptation
    """
    sum_on = sum_off = 0.0
    num_on = num_off = 0

    for i in range(int((-window_size / 2)), int((window_size / 2)) + 1, 1):
        for j in range(int((-window_size / 2)), int((window_size / 2)) + 1, 1):
            if image[row + i][col + j]:
                sum_on += original_image[row + i][col + j]
                num_on += 1
            else:
                sum_off += original_image[row + i][col + j]
                num_off += 1

    if sum_off > 0:
        avg_off = sum_off / num_off
    else:
        avg_off = 0

    if sum_on > 0:
        avg_on = sum_on / num_on
    else:
        avg_on = 0

    return avg_off - avg_on


@jit(nopython=True)
def locate_zero_crossings(original, smoothed_image, laplace_image, win_size, outline=0):
    """
    At the location of an edge pixel there will be a zero crossing in the second derivative of the filtered image.
    This means that the gradient at that point is either a maximum or a minimum. If the second derivative changes sign from
    positive to negative, this is called a positive zero crossing, and if it changes from negative to positive, it is called
    a negative zero crossing. We will allow positive zero crossings to have a positive gradient, and negative zero crossings
    to have a negative gradient. All other zero crossings are assumed to be false (spurious) and are not considered to
    correspond to an edge.
    :param original: original image
    :param smoothed_image: isef_filter output
    :param laplace_image: binary laplace output
    :param win_size: size of adaptive gradient mask
    :param outline: border of image
    :return: zero crossing image
    """
    image = np.zeros(original.shape)

    for row in range(int(win_size / 2) + 1, original.shape[0] - int(win_size / 2), 1):
        for col in range(int(win_size / 2) + 1, original.shape[1] - int(win_size / 2), 1):
            if row < 0 or row >= original.shape[0] or col < 0 or col >= original.shape[1]:
                continue
            elif is_candidate_edge(laplace_image, smoothed_image, row, col):
                image[row][col] = compute_adaptive_gradient(laplace_image, smoothed_image, win_size, row, col)

    return image


@jit(nopython=True)
def estimate_thresh(laplace, ratio):
    vmin = laplace.min()
    vmax = laplace.max()
    hist = np.zeros(256)
    scale = 256.0 / (vmax - vmin + 1)

    for row in range(0, laplace.shape[0], 1):
        for col in range(0, laplace.shape[1], 1):
            hist[int((laplace[row][col] - vmin) * scale)] += 1

    k = 255
    j = int(ratio * laplace.shape[0] * laplace.shape[1])
    count = hist[255]

    while count < j:
        k -= 1
        if k < 0:
            break
        count += hist[k]

    hi = int(k / scale + vmin)

    return hi, int(hi / 2)


@jit(nopython=True)
def mark_connected(edge, laplace, i, j, level, low_thresh, thin_factor):
    # stop if you go off the edge of the image
    if i >= laplace.shape[0] or i < 0 or j >= laplace.shape[1] or j < 0:
        return 0

    # stop if the point has already been visited
    if edge[i][j] != 0:
        return 0

    # stop when you hit an image boundary
    if laplace[i][j] == 0.0:
        return 0

    if laplace[i][j] > low_thresh:
        edge[i][j] = 1
    else:
        edge[i][j] = 255

    not_chain_end = 0
    not_chain_end |= mark_connected(edge, laplace, i, j + 1, level + 1, low_thresh, thin_factor=thin_factor)
    not_chain_end |= mark_connected(edge, laplace, i, j - 1, level + 1, low_thresh, thin_factor=thin_factor)
    not_chain_end |= mark_connected(edge, laplace, i + 1, j + 1, level + 1, low_thresh, thin_factor=thin_factor)
    not_chain_end |= mark_connected(edge, laplace, i + 1, j, level + 1, low_thresh, thin_factor=thin_factor)
    not_chain_end |= mark_connected(edge, laplace, i + 1, j - 1, level + 1, low_thresh, thin_factor=thin_factor)
    not_chain_end |= mark_connected(edge, laplace, i - 1, j - 1, level + 1, low_thresh, thin_factor=thin_factor)
    not_chain_end |= mark_connected(edge, laplace, i - 1, j, level + 1, low_thresh, thin_factor=thin_factor)
    not_chain_end |= mark_connected(edge, laplace, i - 1, j + 1, level + 1, low_thresh, thin_factor=thin_factor)

    if not_chain_end and level > 0:
        if thin_factor > 0:
            if level % thin_factor != 0:
                edge[i][j] = 255

    return 1


@jit(nopython=True)
def threshold_edges(laplace, ratio, thin_factor):
    high_thresh, low_thresh = estimate_thresh(laplace=laplace, ratio=ratio)

    edges = np.zeros(laplace.shape)

    for row in range(0, laplace.shape[0], 1):
        for col in range(0, laplace.shape[1], 1):
            # only check a contour if it is above high_thresh
            if laplace[row][col] > high_thresh:
                # mark all connected points above low thresh
                mark_connected(edge=edges, laplace=laplace, i=row, j=col, level=0, low_thresh=low_thresh, thin_factor=thin_factor)

    for row in range(laplace.shape[0]):
        for col in range(laplace.shape[1]):
            if edges[row][col] == 255:
                edges[row][col] = 0

    return edges


def main_isef_smoothing(param_list: list = None) -> bool:
    """
    Main function for  Infinite Symmetric Exponential Filter (ISEF).
    The ISEF will filtering out any noise that will affect the quality of the image.
    :param param_list: Param needed to respect the following list:
                       [port_in name, wave_offset, smoothing_factor, port_out_name]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_INPUT = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_B_VALUE = 2
    # noinspection PyPep8Naming
    PORT_OUT_POS = 3

    # verify that the number of parameters are OK.
    if len(param_list) != 4:
        log_error_to_console("ISEF FILTER JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_INPUT], wave_offset=param_list[PORT_IN_WAVE])
        port_out = get_port_from_wave(name=param_list[PORT_OUT_POS])
        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                port_out.arr[:] = isef_filter(image=port_in.arr, smoothing_factor=param_list[PORT_B_VALUE])
                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("ISEF FILTER JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_zero_crossing_adaptive(param_list: list = None) -> bool:
    """
    Main function for zero crossing with adaptiv window.
    The ISEF will filtering out any noise that will affect the quality of the image.
    :param param_list: Param needed to respect the following list:
                       [port_in name: input_image, bli_image, isef_image,    wave_offset,
                        port_win_size: what size the mask should be
                        port_out_name name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_INPUT = 0
    # noinspection PyPep8Naming
    PORT_ISEF_INPUT = 1
    # noinspection PyPep8Naming
    PORT_LAPLACE_INPUT = 2
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 3
    # noinspection PyPep8Naming
    PORT_IN_WIN_SIZE = 4
    # noinspection PyPep8Naming
    PORT_OUT_POS = 5

    # verify that the number of parameters are OK.
    if len(param_list) != 6:
        log_error_to_console("ISEF FILTER JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in_original = get_port_from_wave(param_list[PORT_INPUT], param_list[PORT_IN_WAVE])
        port_in_isef = get_port_from_wave(param_list[PORT_ISEF_INPUT], param_list[PORT_IN_WAVE])
        port_in_laplace = get_port_from_wave(param_list[PORT_LAPLACE_INPUT], param_list[PORT_IN_WAVE])
        port_out = get_port_from_wave(param_list[PORT_OUT_POS])

        # check if port's you want to use are valid
        if port_in_original.is_valid() is True and port_in_isef.is_valid() is True and port_in_laplace.is_valid() is True:
            try:
                port_out.arr[:] = locate_zero_crossings(original=port_in_original.arr, smoothed_image=port_in_isef.arr,
                                                        laplace_image=port_in_laplace.arr, win_size=param_list[PORT_IN_WIN_SIZE])
                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("ISEF FILTER JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_thr_hysteresis(param_list: list = None) -> bool:
    """
    Main function for zero crossing with adaptiv window.
    The ISEF will filtering out any noise that will affect the quality of the image.
    :param param_list: Param needed to respect the following list:
                       [port_in name: input_image, bli_image, isef_image,    wave_offset,
                        port_win_size: what size the mask should be
                        port_out_name name: str]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_ZC_INPUT = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_RATIO = 2
    # noinspection PyPep8Naming
    PORT_IN_THINNING = 3
    # noinspection PyPep8Naming
    PORT_OUT_POS = 4

    # verify that the number of parameters are OK.
    if len(param_list) != 5:
        log_error_to_console("THRESHOLD HYSTERESIS ISEF FILTER JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(param_list[PORT_ZC_INPUT], param_list[PORT_IN_WAVE])
        port_out = get_port_from_wave(param_list[PORT_OUT_POS])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                result = threshold_edges(laplace=port_in.arr, ratio=param_list[PORT_IN_RATIO], thin_factor=param_list[PORT_IN_THINNING])
                port_out.arr[:] = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("ISEF FILTER JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


if __name__ == "__main__":
    pass
