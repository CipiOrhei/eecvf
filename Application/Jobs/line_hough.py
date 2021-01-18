from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_file, log_error_to_console

# noinspection PyPackageRequirements
import cv2
import numpy as np
import math


# define a init function, function that will be executed at the begging of the wave
def init_func_global() -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :return: INIT or NOT_INIT state for the job
    """
    log_to_file('Shape Detected Hough')
    return JobInitStateReturn(True)


# define a main function, function that will be executed at the begging of the wave
def main_func_hough(param_list: list = None) -> bool:
    """
    The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure.
    This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a
    so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform.
    The classical Hough transform was concerned with the identification of lines in the image
    :param param_list: Param needed to respect the following list:
                       [enumerate list]
    :return: True if the job executed OK.
    """
    # variables for position of param needed
    # ex:
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_RHO = 2
    # noinspection PyPep8Naming
    PORT_IN_THETA = 3
    # noinspection PyPep8Naming
    PORT_IN_THR = 4
    # noinspection PyPep8Naming
    PORT_IN_MIN_THETA = 5
    # noinspection PyPep8Naming
    PORT_IN_MAX_THETA = 6
    # noinspection PyPep8Naming
    PORT_IN_MIN_LINE_LENGTH = 7
    # noinspection PyPep8Naming
    PORT_IN_MAX_LINE_GAP = 8
    # noinspection PyPep8Naming
    PORT_OUT_POS_IMG = 9
    # noinspection PyPep8Naming
    PORT_OUT_POS_ARRAY = 10
    # noinspection PyPep8Naming
    PORT_OUT_OVERLAY = 11

    # verify that the number of parameters are OK.
    if len(param_list) != 12:
        log_error_to_console("HOUGH TRANSFORM JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_out_img = get_port_from_wave(name=param_list[PORT_OUT_POS_IMG])
        port_out_array = get_port_from_wave(name=param_list[PORT_OUT_POS_ARRAY])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                if param_list[PORT_OUT_OVERLAY] is False:
                    output = np.zeros(port_in.arr.shape)
                    line_color = 255
                else:
                    output = np.zeros((port_in.arr.shape[0], port_in.arr.shape[1], 3), dtype=port_in.arr.dtype)
                    output[:, :, 0] = port_in.arr.copy()
                    output[:, :, 1] = port_in.arr.copy()
                    output[:, :, 2] = port_in.arr.copy()
                    line_color = (0, 0, 255)

                if param_list[PORT_IN_MIN_LINE_LENGTH] == 0 and param_list[PORT_IN_MAX_LINE_GAP] == 0:
                    lines = cv2.HoughLines(image=port_in.arr.copy(), rho=param_list[PORT_IN_RHO], theta=param_list[PORT_IN_THETA],
                                           threshold=param_list[PORT_IN_THR], min_theta=param_list[PORT_IN_MIN_THETA],
                                           max_theta=param_list[PORT_IN_MAX_THETA])

                    if lines is not None:
                        for i in range(0, len(lines)):
                            rho = lines[i][0][0]
                            theta = lines[i][0][1]
                            a = math.cos(theta)
                            b = math.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                            cv2.line(output, pt1, pt2, line_color, 1, cv2.LINE_AA)
                            port_out_array.arr[i][:] = (pt1[0], pt1[1], pt2[0], pt2[1])
                else:
                    lines = cv2.HoughLinesP(image=port_in.arr.copy(),
                                            rho=param_list[PORT_IN_RHO], theta=param_list[PORT_IN_THETA],
                                            threshold=param_list[PORT_IN_THR],
                                            minLineLength=param_list[PORT_IN_MIN_LINE_LENGTH],
                                            maxLineGap=param_list[PORT_IN_MAX_LINE_GAP])

                    if lines is not None:
                        for i in range(0, len(lines)):
                            cv2.line(img=output, pt1=(lines[i][0][0], lines[i][0][1]), pt2=(lines[i][0][2], lines[i][0][3]),
                                     color=line_color, thickness=1, lineType=cv2.LINE_AA)
                            port_out_array.arr[i][:] = lines[i][:]

                port_out_img.arr[:] = output

                log_to_file(str(len(lines)))

                port_out_img.set_valid()
                port_out_array.set_valid()
            except BaseException as error:
                log_error_to_console("HOUGH TRANSFORM JOB NOK: ", str(error))
                log_to_file('')
                pass
        else:
            return False

        return True


def main_func_hough_circle(param_list: list = None) -> bool:
    """
    The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure.
    This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a
    so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform.
    The classical Hough transform was concerned with the identification of lines in the image
    :param param_list: Param needed to respect the following list:
                       [enumerate list]
    :return: True if the job executed OK.
    """
    # variables for position of param needed
    # ex:
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_MIN_DIST = 4
    # noinspection PyPep8Naming
    PORT_IN_MIN_RADIUS = 7
    # noinspection PyPep8Naming
    PORT_IN_MAX_RADIUS = 8
    # noinspection PyPep8Naming
    PORT_IN_METHOD = 2
    # noinspection PyPep8Naming
    PORT_IN_DP = 3
    # noinspection PyPep8Naming
    PORT_IN_PARAM_1 = 5
    # noinspection PyPep8Naming
    PORT_IN_PARAM_2 = 6
    # noinspection PyPep8Naming
    PORT_OUT_POS_IMG = 9
    # noinspection PyPep8Naming
    PORT_OUT_POS_ARRAY = 10
    # noinspection PyPep8Naming
    PORT_OUT_OVERLAY = 11

    # verify that the number of parameters are OK.
    if len(param_list) != 12:
        log_error_to_console("HOUGH CIRCLE TRANSFORM JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_out_img = get_port_from_wave(name=param_list[PORT_OUT_POS_IMG])
        port_out_array = get_port_from_wave(name=param_list[PORT_OUT_POS_ARRAY])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            # try:
            if True:
                if param_list[PORT_OUT_OVERLAY] is False:
                    output = np.zeros(port_in.arr.shape)
                    line_color = 255
                else:
                    output = np.zeros((port_in.arr.shape[0], port_in.arr.shape[1], 3), dtype=port_in.arr.dtype)
                    output[:, :, 0] = port_in.arr.copy()
                    output[:, :, 1] = port_in.arr.copy()
                    output[:, :, 2] = port_in.arr.copy()
                    line_color = (0, 0, 255)

                circles = cv2.HoughCircles(image=port_in.arr.copy(),
                                           method=param_list[PORT_IN_METHOD], dp=param_list[PORT_IN_DP],
                                           minDist=param_list[PORT_IN_MIN_DIST], param1=param_list[PORT_IN_PARAM_1],
                                           param2=param_list[PORT_IN_PARAM_2], minRadius=param_list[PORT_IN_MIN_RADIUS],
                                           maxRadius=param_list[PORT_IN_MAX_RADIUS])

                circles = np.uint16(np.around(circles))
                print(circles)
                if circles is not None:
                    for i in range(0, len(circles[0])):
                        # draw the outer circle
                        cv2.circle(img=output, center=(circles[0][i][0], circles[0][i][1]), radius=circles[0][i][2], color=line_color, thickness=2)
                        # draw the center of the circle
                        cv2.circle(img=output, center=(circles[0][i][0], circles[0][i][1]), radius=2, color=line_color, thickness=3)

                        port_out_array.arr[i][:] = circles[0][i][:]

                port_out_img.arr[:] = output

                log_to_file(str(len(circles)))

                port_out_img.set_valid()
                port_out_array.set_valid()
            # except BaseException as error:
            #     log_error_to_console("HOUGH TRANSFORM JOB NOK: ", str(error))
            #     log_to_file('')
            #     pass
        else:
            return False

        return True


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
