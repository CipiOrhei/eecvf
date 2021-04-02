# Do not delete used indirectly
from typing import Tuple
# noinspection PyUnresolvedReferences
from Application.Frame import transferJobPorts
from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console, log_to_console
from Application.Config.create_config import jobs_dict, create_dictionary_element
import config_main
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file

import tensorflow as tf
import numpy as np
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from Application.Jobs.external.Deep_Video_Deinterlacing.TSNet import TSNet

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# global variables
x = None
y = None
z = None
sess = None
model = None

############################################################################################################################################
# Init functions
############################################################################################################################################


def init_func() -> JobInitStateReturn:
    """
    Init function for the Deep Video Deinterlacing algorithm
    :return: INIT or NOT_INIT state for the job
    """
    global x, y, z, sess, model
    location_model = "Application/Jobs/external/Deep_Video_Deinterlacing/models/TSNet_advanced.model"

    try:
        image_test = np.zeros((global_var_handler.HEIGHT_L0, global_var_handler.WIDTH_L0, 3))
        input_image = np.swapaxes(np.swapaxes(image_test, 0, 2), 1, 2)
        input_image = input_image.reshape((3, global_var_handler.HEIGHT_L0, global_var_handler.WIDTH_L0, 1))

        tf.device('/gpu:0')
        model = TSNet()
        x = tf.placeholder(tf.float32, shape=[3, global_var_handler.HEIGHT_L0, global_var_handler.WIDTH_L0, 1])
        y, z, y_full, z_full = model.createNet(x)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        print(os.getcwd())
        saver = tf.train.Saver()
        saver.restore(sess, location_model)
        lower, upper = sess.run([y, z], feed_dict={x: input_image})

    except Exception as ex:
        log_error_to_console('INIT DEEP VIDEO DEINTERLACING NOK', ex.__str__())
        return JobInitStateReturn(False)

    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################


def main_func(param_list: list = None) -> bool:
    """
    Main function for Deep Video Deinterlacing calculation job.
    :param param_list: Param needed to respect the following list:
                       [input_image_name, input_wave, output_image_field_0, output_image_field_1]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_OUT_FRAME_0 = 2
    # noinspection PyPep8Naming
    PORT_OUT_FRAME_1 = 3
    # verify that the number of parameters are OK.
    if len(param_list) != 4:
        log_error_to_console("DEEP VIDEO DEINTERLACING JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_out_frame_0 = get_port_from_wave(name=param_list[PORT_OUT_FRAME_0], wave_offset=0)
        port_out_frame_1 = get_port_from_wave(name=param_list[PORT_OUT_FRAME_1], wave_offset=0)

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
                image = port_in.arr.copy()
                image = image.astype('float32') / 255.
                img_height, img_width, img_nchannels = image.shape

                input_image = np.swapaxes(np.swapaxes(image, 0, 2), 1, 2)
                input_image = input_image.reshape((3, img_height, img_width, 1))
                im1 = np.zeros(image.shape).astype('float32')
                im2 = np.zeros(image.shape).astype('float32')
                im1[0::2, :, :] = image[0::2, :, :]
                im2[1::2, :, :] = image[1::2, :, :]

                lower, upper = sess.run([y, z], feed_dict={x: input_image})
                lower_Field = np.swapaxes(np.swapaxes(lower, 1, 2), 0, 2)
                upper_Field = np.swapaxes(np.swapaxes(upper, 1, 2), 0, 2)
                im1[1::2, :, :] = lower_Field.reshape((int(img_height / 2), img_width, 3))
                im2[0::2, :, :] = upper_Field.reshape((int(img_height / 2), img_width, 3))
                im1 = im1.astype(np.float32) * 255.0
                im1 = np.clip(im1, 0, 255).astype('uint8')
                im2 = im2.astype(np.float32) * 255.0
                im2 = np.clip(im2, 0, 255).astype('uint8')

                port_out_frame_0.arr[:] = im1
                port_out_frame_0.set_valid()
                port_out_frame_1.arr[:] = im2
                port_out_frame_1.set_valid()

            except BaseException as error:
                log_error_to_console("DEEP VIDEO DEINTERLACING JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


############################################################################################################################################
# Job create functions
############################################################################################################################################

def do_deep_video_deinterlacing(port_input_name: str, wave_offset: int = 0,
                                port_output_name: str = None, is_rgb=True,
                                level: config_main.PYRAMID_LEVEL = config_main.PYRAMID_LEVEL.LEVEL_0) -> Tuple[str, str]:
    """
     This job runs a specific deinterlacing network (DIN), which is motivated by the traditional deinterlacing strategy.
     The proposed DIN consists of two stages, i.e., a cooperative vertical interpolation stage for split fields, and a merging stage that
     is applied to perceive movements and remove ghost artifacts. Experimental results demonstrate that the proposed method can effectively
     remove complex artifacts in early interlaced videos.
     https://arxiv.org/abs/2011.13675
     :param port_input_name: name of input port
     :param port_output_name: name of output port
     :param is_rgb: if input port and output port is rgb
     :param level: pyramid level to calculate at
     :param wave_offset: port wave offset. If 0 it is in current wave.
     :return: output image port names: port_edges_name_output, port_edge_map_name_output, port_lines_name_output, port_lines_img_output
     """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'DEEP_DEINTERLACE_FRAME'

    port_output_name_frame_0 = port_output_name + '_0'
    port_output_name_frame_1 = port_output_name + '_1'

    output_port_name_frame_0 = transform_port_name_lvl(name=port_output_name_frame_0, lvl=level)
    output_port_name_frame_0_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    output_port_name_frame_1 = transform_port_name_lvl(name=port_output_name_frame_1, lvl=level)
    output_port_name_frame_1_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, output_port_name_frame_0, output_port_name_frame_1]
    output_port_list = [(output_port_name_frame_0, output_port_name_frame_0_size, 'B', True),
                        (output_port_name_frame_1, output_port_name_frame_1_size, 'B', True)]

    job_name = job_name_create(action='Deep Video Deinterlacing', input_list=input_port_list, wave_offset=[wave_offset], level=level)
    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name_frame_0, port_output_name_frame_1


if __name__ == "__main__":
    pass
