# noinspection PyPackageRequirements
import cv2
import numpy as np
import skimage.transform as trans

# noinspection PyUnresolvedReferences
from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console

from MachineLearning.Models.unet_edge import unet

model = None

"""
Module handles u-net edge detection image jobs for the APPL block.
"""


def init_func(param_list) -> JobInitStateReturn:
    """
    Init function for the job
    :param param_list: Param needed list of port names [model]
    :return: INIT or NOT_INIT state for the job
    """
    global model

    location_model = 'MachineLearning/model_weights/' + param_list[0] + '.hdf5'
    try:
        model = unet(pretrained_weights=location_model)

        img = trans.resize(image=np.zeros(shape=(512, 512)), output_shape=(512, 512))
        img = np.reshape(a=img, newshape=img.shape + (1,))
        img = np.reshape(a=img, newshape=(1,) + img.shape)
        # noinspection PyUnusedLocal
        img_out = model.predict(x=img, verbose=0)
    except Exception as ex:
        log_error_to_console('INIT U-NET NOK', ex.__str__())
        return JobInitStateReturn(False)

    return JobInitStateReturn(True)


def main_run_unet_edge_func(port_list: list = None) -> bool:
    """
    Main function for edge detection using u-net model

    :param port_list: Param needed list of port names [input, output]
                      List of ports passed as parameters should be even. Every input picture should have a output port.
    :return: True if the job executed OK.
    """
    # check if param OK
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE_IMG = 1
    # noinspection PyPep8Naming
    PORT_OUT_POS = 2

    if len(port_list) < 3:
        log_error_to_console("U-NET EDGE PARAM NOK", str(len(port_list)))
        return False
    else:
        global model

        p_in = get_port_from_wave(name=port_list[PORT_IN_POS], wave_offset=port_list[PORT_IN_WAVE_IMG])
        p_out = get_port_from_wave(name=port_list[PORT_OUT_POS])

        if p_in.is_valid() is True and model is not None:
            try:
                level = p_out.name.split('_')[-1]
                w = eval('global_var_handler.WIDTH_' + level)
                h = eval('global_var_handler.HEIGHT_' + level)

                img = np.array(p_in.arr) / 255
                img = trans.resize(img, (512, 512))
                img = np.reshape(img, img.shape + (1,))
                img = np.reshape(img, (1,) + img.shape)

                # noinspection PyUnresolvedReferences
                img_out = model.predict(x=img, verbose=0)
                img_out = img_out[0].reshape((512, 512))
                img_out = img_out * 255
                img_out = cv2.normalize(img_out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

                p_out.arr[:] = cv2.resize(src=img_out, dsize=(w, h), interpolation=cv2.INTER_AREA)
                p_out.set_valid()
            except BaseException as error:
                log_error_to_console("U-NET EDGE JOB NOK: ", str(error))
                pass
        else:
            return False

    return True
