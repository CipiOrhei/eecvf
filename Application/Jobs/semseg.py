# noinspection PyUnresolvedReferences
import os
import cv2
import numpy as np

from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console, log_setup_info_to_console, log_to_console

from MachineLearning.external.image_segmentation_keras.keras_segmentation.predict import model_from_checkpoint_path
from MachineLearning.external.image_segmentation_keras.keras_segmentation.data_utils.data_loader import get_image_array
from MachineLearning.external.image_segmentation_keras.keras_segmentation.predict import predict, visualize_segmentation

model_semseg = dict()

"""
Module handles u-net edge detection image jobs for the APPL block.
"""


def init_func_semseg_keras_repo(param_list) -> JobInitStateReturn:
    """
    Init function for the job vgg_unet model
    :param param_list: Param needed list of port names [model]
    :return: INIT or NOT_INIT state for the job
    """
    # noinspection PyPep8Naming
    PORT_LOCATION_MODEL = 0
    # noinspection PyPep8Naming
    PORT_MODEL = 1
    # noinspection PyPep8Naming
    PORT_OUTPUT_PORT_SIZE = 2

    if not os.path.exists(param_list[PORT_LOCATION_MODEL]):
        log_error_to_console('INIT' + param_list[PORT_MODEL].upper() + 'NOK','model not in location')

    try:
        global model_semseg
        #
        path = os.path.join(os.getcwd(),param_list[PORT_LOCATION_MODEL], param_list[PORT_MODEL])
        model_semseg[param_list[PORT_MODEL]] = model_from_checkpoint_path(checkpoints_path=path)
        size = eval('global_var_handler.' + param_list[PORT_OUTPUT_PORT_SIZE] + '_RGB')
        img = np.zeros(size)
        # noinspection PyUnusedLocal
        img_out = predict(model=model_semseg[param_list[PORT_MODEL]], inp=img)
    except Exception as ex:
        log_error_to_console('INIT' + param_list[PORT_MODEL].upper() + 'NOK', ex.__str__())
        return JobInitStateReturn(False)

    return JobInitStateReturn(True)


def main_func_semseg_keras_repo(param_list: list = None) -> bool:
    """
    Main function for {job} calculation job.
    :param param_list: Param needed to respect the following list:
                       [enumerate list]
    :return: True if the job executed OK.
    """
    # variables for position of param needed
    # ex:

    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_OUT_POS = 2
    # noinspection PyPep8Naming
    PORT_SAVE_AUGMENTATION = 3
    # noinspection PyPep8Naming
    PORT_NR_CLASSES = 4
    # noinspection PyPep8Naming
    PORT_OVERLAY = 5
    # noinspection PyPep8Naming
    PORT_SHOW_LEGEND = 6
    # noinspection PyPep8Naming
    PORT_CLASS_COLOR_LIST = 7
    # noinspection PyPep8Naming
    PORT_CLASS_NAME_LIST = 8
    # noinspection PyPep8Naming
    PORT_MODEL_NAME_LIST = 9
    # noinspection PyPep8Naming
    PORT_OUT_OVERLAY = 10

    # verify that the number of parameters are OK.
    if len(param_list) != 11:
        log_error_to_console("SEMSEG" + param_list[PORT_MODEL_NAME_LIST] + "JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        global model_semseg

        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE_IMG])
        port_out = get_port_from_wave(name=param_list[PORT_OUT_POS])

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                level = port_out.name.split('_')[-1]
                w = eval('global_var_handler.WIDTH_' + level)
                h = eval('global_var_handler.HEIGHT_' + level)

                img_out = predict(model=model_semseg[param_list[PORT_MODEL_NAME_LIST]], inp=port_in.arr.copy())

                if param_list[PORT_SAVE_AUGMENTATION] is True:
                    port_in_overlay = get_port_from_wave(name=param_list[PORT_OUT_OVERLAY], wave_offset=param_list[PORT_IN_WAVE_IMG])

                    if param_list[PORT_SHOW_LEGEND] is True and port_in.arr.shape == port_in_overlay.arr.shape:
                        new_shape = list(port_in_overlay.arr.shape)
                        new_shape[1] = new_shape[1] + 125
                        new_shape = tuple(new_shape)
                        port_in_overlay.arr = np.zeros(tuple(new_shape))

                    if not param_list[PORT_CLASS_COLOR_LIST]:
                        port_in_overlay.arr[:] = visualize_segmentation(seg_arr=img_out, inp_img=port_in.arr.copy(),
                                                                        n_classes=param_list[PORT_NR_CLASSES],
                                                                        overlay_img=param_list[PORT_OVERLAY],
                                                                        show_legends=param_list[PORT_SHOW_LEGEND],
                                                                        class_names=param_list[PORT_CLASS_NAME_LIST],
                                                                        prediction_width=w, prediction_height=h)
                    else:
                        port_in_overlay.arr[:] = visualize_segmentation(seg_arr=img_out, inp_img=port_in.arr.copy(),
                                                                        n_classes=param_list[PORT_NR_CLASSES],
                                                                        overlay_img=param_list[PORT_OVERLAY],
                                                                        colors=param_list[PORT_CLASS_COLOR_LIST],
                                                                        class_names=param_list[PORT_CLASS_NAME_LIST],
                                                                        show_legends=param_list[PORT_SHOW_LEGEND],
                                                                        prediction_width=w, prediction_height=h)
                    port_in_overlay.set_valid()

                img_out = cv2.resize(src=img_out, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
                port_out.arr[:] = img_out
                port_out.set_valid()

            except BaseException as error:
                log_error_to_console("VGG-U-UNET JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


if __name__ == "__main__":
    pass