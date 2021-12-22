# noinspection PyPackageRequirements
import cv2

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import MORPH_CONFIG, PYRAMID_LEVEL, FILTERS, CANNY_VARIANTS, FILTERS_SECOND_ORDER, THRESHOLD_CONFIG
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create

# for using same names for levels
from Utils.log_handler import log_to_console, log_setup_info_to_console, log_error_to_console

import numpy as np

# TODO change handling of port type so it can be configurable

"""
All the jobs from this block have an wrapper in this module. This module handles the transition between the user interface and 
jobs of the EECVF.
All functions from this module will add one or more jobs to the job buffer. 
"""

# list to hold custom kernels used in application
# important to avoid confusion of names
custom_kernels_used = []

############################################################################################################################################
# Input jobs
############################################################################################################################################

# TODO create possibility to add multiple input folders


def do_get_image_job(port_output_name: str = 'RAW', direct_grey: bool = False) -> str:
    """
    Function for configure the image retrieval job from folder.
    The job is added to the job buffer.
    :param port_output_name: name you want to use for raw image in the application
    :param direct_grey: If we want to get the image direct grey values
    :return: output image port name
    """
    output_raw_port_name = transform_port_name_lvl(name=port_output_name, lvl=PYRAMID_LEVEL.LEVEL_0)

    if direct_grey is True:
        output_raw_port_size = transform_port_size_lvl(lvl=PYRAMID_LEVEL.LEVEL_0, rgb=False)
    else:
        output_raw_port_size = transform_port_size_lvl(lvl=PYRAMID_LEVEL.LEVEL_0, rgb=True)

    input_port_list = None
    main_func_list = [output_raw_port_name, direct_grey]
    output_port_list = [(output_raw_port_name, output_raw_port_size, 'B', True)]

    job_name = job_name_create(action='Get image frame')

    d = create_dictionary_element(job_module='get_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_get_satellite_image_job(port_output_name: str = 'RAW'):
    """
    Function for configure the satellite jpeg2000 image retrieval job from folder.
    The job is added to the job buffer.
    :param port_output_name: name you want to use for raw image in the application
    :return: None
    """

    output_raw_port_name = transform_port_name_lvl(name=port_output_name, lvl=PYRAMID_LEVEL.LEVEL_0)
    output_raw_port_size = transform_port_size_lvl(lvl=PYRAMID_LEVEL.LEVEL_0, rgb=True)

    job_name = job_name_create(action='Get satellite image frame')

    d = create_dictionary_element(job_module='get_image',
                                  job_name=job_name,
                                  input_ports=None,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_satellite',
                                  main_func_param=[output_raw_port_name],
                                  output_ports=[(output_raw_port_name, output_raw_port_size, 'B')])

    jobs_dict.append(d)


def do_get_video_job(port_output_name: str = 'RAW'):
    """
    Function for configure the image retrieval job from video input.
    :param port_output_name: name you want to use for raw image in the application
    :return: output image port name
    """
    output_raw_port_name = transform_port_name_lvl(name=port_output_name, lvl=PYRAMID_LEVEL.LEVEL_0)
    output_raw_port_size = transform_port_size_lvl(lvl=PYRAMID_LEVEL.LEVEL_0, rgb=True)

    input_port_list = None
    main_func_list = [output_raw_port_name]
    output_port_list = [(output_raw_port_name, output_raw_port_size, 'B', True)]

    job_name = job_name_create(action='Get image video frame')

    d = create_dictionary_element(job_module='get_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_video',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_get_video_capture_job(port_output_name: str = 'RAW') -> str:
    """
    Function for configure the image retrieval job from video camera.
    :param port_output_name: name you want to use for raw image in the application
    :return: output image port name
    """
    output_raw_port_name = transform_port_name_lvl(name=port_output_name, lvl=PYRAMID_LEVEL.LEVEL_0)
    output_raw_port_size = transform_port_size_lvl(lvl=PYRAMID_LEVEL.LEVEL_0, rgb=True)

    input_port_list = None
    main_func_list = [output_raw_port_name]
    output_port_list = [(output_raw_port_name, output_raw_port_size, 'B', True)]

    job_name = job_name_create(action='Get image camera video frame')

    d = create_dictionary_element(job_module='get_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_video_camera',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)
    return port_output_name


def do_get_image_from_txt_job(line_separator, pixel_separator,
                              port_output_name: str = 'RAW', is_rgb=False) -> str:
    """
    Function for configure the image retrieval job from folder.
    The job is added to the job buffer.
    :param line_separator: character used to separate lines
    :param pixel_separator: character used to separate pixels
    :param port_output_name: name you want to use for raw image in the application
    :param is_rgb: if resulting image is RGB
    :return: output image port name
    """
    output_raw_port_name = transform_port_name_lvl(name=port_output_name, lvl=PYRAMID_LEVEL.LEVEL_0)
    output_raw_port_size = transform_port_size_lvl(lvl=PYRAMID_LEVEL.LEVEL_0, rgb=is_rgb)

    input_port_list = None
    main_func_list = [output_raw_port_name]
    output_port_list = [(output_raw_port_name, output_raw_port_size, 'B', True)]

    if isinstance(line_separator, str):
        main_func_list.append(line_separator)
    else:
        log_error_to_console("LINE SEPARATOR IS NOT STRING")

    if isinstance(pixel_separator, str):
        main_func_list.append(pixel_separator)
    else:
        log_error_to_console("PIXEL SEPARATOR IS NOT STRING")

    job_name = job_name_create(action='Get image frame')

    d = create_dictionary_element(job_module='get_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_from_txt',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


############################################################################################################################################
# Image processing jobs
############################################################################################################################################


def do_grayscale_transform_job(port_input_name: str,
                               port_output_name: str = None,
                               level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Function for transforming RGB image to grayscale image. The implementation is done using OpenCV-image library.
    :param port_input_name: name of port name that contains a RGB image
    :param port_output_name: name you want to use for output port
    :param level: on what pyramid level of image you want this job to run
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """
    input_raw_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'GRAY_' + port_input_name

    output_raw_grey_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_raw_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_raw_port_name]
    main_func_list = [input_raw_port_name, wave_offset, output_raw_grey_port_name]
    output_port_list = [(output_raw_grey_port_name, output_raw_port_size, 'B', True)]

    job_name = job_name_create(action='Greyscale transform', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='transform_to_greyscale',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_image_complement_job(port_input_name: str,
                            port_output_name: str = None,
                            level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    Returns the image complement or invert of the input image. 255 - pixel value.
    :param port_input_name: name of input port
    :param port_output_name: name of output port to use
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :param level: On what pyramid level of image you want this job to run
    :return: output image port name
    """
    input_raw_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'COMPLEMENT_' + port_input_name

    output_raw_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_raw_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_raw_port_name]
    main_func_list = [input_raw_port_name, wave_offset, output_raw_port_name]
    output_port_list = [(output_raw_port_name, output_raw_port_size, 'B', True)]

    job_name = job_name_create(action='Complement', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='image_complement',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_number_edge_pixels(port_input_name: str,
                          port_output_name: str = None,
                          wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> None:
    """
    Returns the number of edge pixels, different from zero, found in image.
    It work only for grayscale images.
    :param port_input_name: name you want to use for raw image in the application
    :param port_output_name: name of output port to use
    :param level: On what pyramid level of image you want this job to run
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: None
    """
    input_raw_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'NR_EDGE_PX_' + port_input_name

    output_raw_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)

    init_param = ['NR EDGE PX ' + input_raw_port_name]

    input_port_list = [input_raw_port_name]
    main_func_list = [input_raw_port_name, wave_offset, output_raw_port_name]
    output_port_list = [(output_raw_port_name, '1', 'L', False)]

    job_name = job_name_create(action='Nr edge px', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=init_param,
                                  main_func_name='nr_edge_px_calculation',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_image_crop_job(port_input_name: str,
                      start_width_percentage: int, end_width_percentage: int, start_height_percentage: int, end_height_percentage: int,
                      port_output_name: str = None, with_resize: bool = False, new_height: int = 0, new_width: int = 0,
                      wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Creates a cropped image according to the specified ratios.
    :param port_input_name: name you want to use for raw image in the application
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param start_width_percentage: percent from the image on the left where the cropped image to start
    :param end_width_percentage: percent from the image on the right where the cropped image to start
    :param start_height_percentage: percent from the image on the top where the cropped image to start
    :param end_height_percentage: percent from the image on the bottom where the cropped image to start
    :param port_output_name: name of output port to use
    :param with_resize: if new image should be resized.
    :param is_rgb: if the output ports is rgb, 3 channels
    :param level: On what pyramid level of image you want this job to run
    :return: output image port name
    """
    input_raw_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'CROPPED_' + port_input_name

    if with_resize is True:
        if is_rgb is True:
            new_size = (new_height, new_width, 3)
        else:
            new_size = (new_height, new_width)

        level = PYRAMID_LEVEL.add_level(size=new_size)

        output_raw_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
        output_raw_port_size = transform_port_size_lvl(lvl=new_size, rgb=is_rgb)
    else:
        output_raw_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
        output_raw_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_raw_port_name]
    main_func_list = [input_raw_port_name, wave_offset, start_width_percentage, end_width_percentage,
                      start_height_percentage, end_height_percentage, with_resize, output_raw_port_name]

    output_port_list = [(output_raw_port_name, output_raw_port_size, 'B', True)]

    size = "[{s_w}%*width:{e_w}%*width][{s_h}%*height:{e_h}%*height]".format(s_w=start_width_percentage, e_w=end_width_percentage,
                                                                             s_h=start_height_percentage, e_h=end_height_percentage)

    job_name = job_name_create(action='Crop', input_list=[input_raw_port_name], wave_offset=[wave_offset], level=level,
                               Size=size)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='do_image_crop',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_add_gaussian_blur_noise_job(port_input_name: str,
                                   port_output_name: str = None, mean_value: float = 0.1, variance: float = 0.1,
                                   wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False) -> str:
    """
    Is statistical noise having a probability density function (PDF) equal to that of the normal distribution, which is also known
    as the Gaussian distribution. The implementation is done using scikit-image library.
    paper/book: https://dl.acm.org/doi/book/10.5555/573190.
    :param port_input_name: name of input port
    :param mean_value: mean of random distribution.
    :param variance: variance of random distribution.
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'GAUSS_NOISE_MEAN_VAL_' + str(mean_value).replace('.', '_') + '_VAR_' + str(variance).replace('.', '_') + \
                           '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]

    main_func_list = [input_port_name, wave_offset, mean_value, variance, output_port_name]

    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Add Gaussian Noise', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               Mean_value=str(mean_value), Variance=str(variance))

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='add_gaussian_noise',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_add_salt_pepper_noise(port_input_name: str,
                             port_output_name: str = None, density: float = 0.3,
                             wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False) -> str:
    """
    Salt-and-pepper noise is a form of noise sometimes seen on images. It is also known as impulse noise.
    This noise can be caused by sharp and sudden disturbances in the image signal. It presents itself as
    sparsely occurring white and black pixels.
    The implementation is done using scikit-image library.
    paper/book: https://dl.acm.org/doi/book/10.5555/573190.
    :param port_input_name: name of input port
    :param port_output_name: name of output port
    :param density: density noise.
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'S&P_NOISE_DENS_' + str(density).replace('.', '_') + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, density, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Add Salt&Pepper Noise', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               Density=str(density))

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='add_salt_pepper_noise',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_add_speckle_noise(port_input_name: str,
                         port_output_name: str = None, variance: float = 5,
                         wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False) -> str:
    """
    Speckle noise is the noise that arises due to the effect of environmental conditions on the imaging sensor during image acquisition.
    The implementation is done using scikit-image library.
    paper/book: https://dl.acm.org/doi/book/10.5555/573190.
    :param variance: variance of random distribution.
    :param port_input_name: name of input port
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'SPACKLE_NOISE_VAR_' + str(variance).replace('.', '_') + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, variance, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Add Speckle Noise', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               Variance=str(variance))

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='add_speckle_noise',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_max_pixel_image_job(port_input_name: str,
                           port_output_name: str = None,
                           wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> None:
    """
    Function for configure the max intensity pixel image calculation job.
    Works only in greyscale images.
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'MAX_PX_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    input_port_list = [input_port_name]
    init_param = ['MAX PX ' + input_port_name]
    main_func_list = [input_port_name, wave_offset, output_port_name]
    output_port_list = [(output_port_name, '1', 'h', False)]

    job_name = job_name_create(action='Max pixel', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=init_param,
                                  main_func_name='max_calculation',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_median_pixel_image_job(port_input_name: str,
                              port_output_name: str = None,
                              wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> None:
    """
    Function for configure the median intensity pixel image calculation job.
    :param port_input_name: name of input port
    :param port_output_name: name of output port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'MEDIAN_PX_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)

    init_param = ['MEDIAN PX ' + input_port_name]
    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, output_port_name]
    output_port_list = [(output_port_name, '1', 'h', False)]

    job_name = job_name_create(action='Median pixel', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=init_param,
                                  main_func_name='median_calculation',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_mean_pixel_image_job(port_input_name: str,
                            port_output_name: str = None,
                            level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> None:
    """
    Function for configure the mean intensity pixel image calculation job.
    :param port_input_name: name of input port
    :param port_output_name: name of output port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'MEAN_PX_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    init_param = ['MEAN PX ' + input_port_name]
    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, output_port_name]
    output_port_list = [(output_port_name, '1', 'h', False)]

    job_name = job_name_create(action='Mean pixel', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=init_param,
                                  main_func_name='mean_calculation',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_rotate_image_job(port_input_name: str,
                        angle: int, reshape: bool, extend_border: bool = False, do_interpolation: bool = True,
                        port_output_name: str = None,
                        level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    Function for rotating an image around the axis.
    :param port_input_name: name of input port
    :param angle: the angle to rotate
    :param reshape: If we want to reshape the image
    :param extend_border: if we want to extend the border pixels
    :param do_interpolation: rotate with interpolation
    :param port_output_name: name of output port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'ROTATE_ANGLE_' + str(angle)
        if reshape:
            port_output_name += '_RESHAPED_'
        if extend_border:
            port_output_name += '_BORDER_EXT_'
        if do_interpolation is False:
            port_output_name += '_NO_INTERPOLATION_'
        port_output_name += port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, angle, reshape, extend_border, do_interpolation, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Rotate', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               ANGLE=str(angle), RESHAPE=reshape, BORDER_EXTENSION=extend_border)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='rotate_main',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_flip_image_job(port_input_name: str,
                      flip_horizontal: bool, flip_vertical: bool,
                      port_output_name: str = None,
                      level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    Function for flipping an image around the axis.
    :param port_input_name: name of input port
    :param flip_horizontal: if we want to flip the image on the x axis
    :param flip_vertical: if we want to flip the image on the y axis
    :param port_output_name: name of output port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :param level: pyramid level to calculate at
    :return: name of output port
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'FLIP_'
        if flip_horizontal:
            port_output_name += 'ON_X_'
        if flip_vertical:
            port_output_name += 'ON_Y_'
        port_output_name += port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, flip_horizontal, flip_vertical, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Flip', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               on_X=str(flip_horizontal), on_Y=flip_vertical)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='flip_main',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_zoom_image_job(port_input_name: str,
                      zoom_factor: float,
                      do_interpolation: bool = True, port_output_name: str = None,
                      level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    Function for zooming an image in or out.
    :param port_input_name: name of input port
    :param zoom_factor: zoom factor
    :param do_interpolation: zoom with interpolation
    :param port_output_name: name of output port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'ZOOM_' + str(zoom_factor) + '_'

        if do_interpolation is False:
            port_output_name += 'NO_INTERPOLATION_'

        port_output_name += port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, zoom_factor, do_interpolation, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Zoom', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               ZOOM=zoom_factor, do_interpolation=do_interpolation)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='zoom_main',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_contrast_brightness_change_image_job(port_input_name: str,
                                            alpha: float = 1.0, beta: int = 0,
                                            port_output_name: str = None,
                                            level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0,
                                            is_rgb: bool = False) -> str:
    """
    Function for zooming an image in or out.
    :param port_input_name: name of input port
    :param alpha: contrast control (1.0-3.0)
    :param beta: Brightness control (0-100)
    :param port_output_name: name of output port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'CHANGE'
        if alpha != 1.0:
            port_output_name += '_ALPHA_' + str(alpha)
        if beta != 0:
            port_output_name += '_BETA_' + str(beta)
        port_output_name += '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, alpha, beta, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Change', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               ALPHA=alpha, BETA=beta)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='change_contrast_brightness_main',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_gamma_correction_image_job(port_input_name: str,
                                  gamma: float,
                                  port_output_name: str = None,
                                  level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    Gamma correction can be used to correct the brightness of an image by using a non linear transformation between the input
    values and the mapped output
    :param port_input_name: name of input port
    :param gamma: gamma 0-250
    :param port_output_name: name of output port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'GAMMA_CORRECTION_' + str(gamma)
        port_output_name += '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, gamma, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Change', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               GAMMA=gamma)

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='gamma_correction_main',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_pixelate_image_job(port_input_name: str,
                          nr_pixels_to_group: int,
                          port_output_name: str = None,
                          level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    Create a pixelate effect on an image.
    :param port_input_name: name of input port
    :param nr_pixels_to_group: number of pixels to group as one pixel
    :param port_output_name: name of output port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'PIXELATE_EFFECT_' + str(nr_pixels_to_group) + '_TO_1'
        port_output_name += '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, nr_pixels_to_group, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Change', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               RATIO=str(nr_pixels_to_group) + '_TO_1')

    d = create_dictionary_element(job_module='processing_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='pixelate_main',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


############################################################################################################################################
# Multiple image jobs
############################################################################################################################################


def do_matrix_difference_job(port_input_name_1: str, port_input_name_2: str,
                             port_output_name: str = None, normalize_image: bool = True, result_is_image: bool = True,
                             wave_offset_port_1: int = 0, wave_offset_port_2: int = 0,
                             is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Matrix difference of 2 images.
    :param port_input_name_1: first matrix
    :param wave_offset_port_1: port wave offset. If 0 it is in current wave.
    :param port_input_name_2: second matrix
    :param wave_offset_port_2: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param normalize_image: if we want the resulting image to be normalized
    :param result_is_image: if the resulted port is an image. If the result is a value set to false.
    :param level:  pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name_1, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_input_name_2, lvl=level)

    if port_output_name is None:
        port_output_name = 'DIFF_' + port_input_name_1 + '_AND_' + port_input_name_2

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_1, input_port_2]
    main_func_list = [input_port_1, wave_offset_port_1, input_port_2, wave_offset_port_2, output_port, normalize_image]
    output_port_list = [(output_port, output_port_size, 'B', result_is_image)]

    job_name = job_name_create(action='Difference', input_list=input_port_list, wave_offset=[wave_offset_port_1, wave_offset_port_2],
                               level=level)

    d = create_dictionary_element(job_module='processing_multiple_images',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=max(wave_offset_port_1, wave_offset_port_2),
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='difference_2_matrix',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_matrix_difference_1_px_offset_job(port_input_name_1: str, port_input_name_2: str,
                                         wave_offset_port_1: int = 0, wave_offset_port_2: int = 0,
                                         port_output_name: str = None,
                                         level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Matrix difference of 2 images with 1 px offset in each direction.
    Works only on grayscale.
    :param port_input_name_1: first matrix
    :param wave_offset_port_1: port wave offset. If 0 it is in current wave.
    :param port_input_name_2: second matrix
    :param wave_offset_port_2: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param level:  pyramid level to calculate at
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name_1, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_input_name_2, lvl=level)

    if port_output_name is None:
        port_output_name = 'DIFF_1_PX_OFFSET_' + port_input_name_1 + '_AND_' + port_input_name_2

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_1, input_port_2]
    main_func_list = [input_port_1, wave_offset_port_1, input_port_2, wave_offset_port_2, output_port]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Difference 1px offset', input_list=input_port_list,
                               wave_offset=[wave_offset_port_1, wave_offset_port_2], level=level)

    d = create_dictionary_element(job_module='processing_multiple_images',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=max(wave_offset_port_1, wave_offset_port_2),
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='difference_2_matrix_1_px_offset',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_matrix_sum_job(port_input_name_1: str, port_input_name_2: str,
                      wave_offset_port_1: int = 0, wave_offset_port_2: int = 0,
                      normalize: bool = False,
                      port_output_name: str = None,
                      is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Matrix sum of 2 images.
    :param port_input_name_1: first matrix
    :param wave_offset_port_1: port wave offset. If 0 it is in current wave.
    :param port_input_name_2: second matrix
    :param wave_offset_port_2: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param normalize: If we want the resulted matrix to be a normalized one of threshold one.
    :param level:  pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name_1, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_input_name_2, lvl=level)

    if port_output_name is None:
        port_output_name = 'SUM_' + port_input_name_1 + '_AND_' + port_input_name_2

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_1, input_port_2]
    main_func_list = [input_port_1, wave_offset_port_1, input_port_2, wave_offset_port_2, output_port, normalize]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Sum', input_list=input_port_list,
                               wave_offset=[wave_offset_port_1, wave_offset_port_2], level=level)

    d = create_dictionary_element(job_module='processing_multiple_images',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=max(wave_offset_port_1, wave_offset_port_2),
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='add_2_matrix',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_matrix_bitwise_and_job(port_input_name_1: str, port_input_name_2: str,
                              wave_offset_port_1: int = 0, wave_offset_port_2: int = 0,
                              port_output_name: str = None,
                              level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Matrix AND for 2 images.
    Works only on grayscale images.
    :param port_input_name_1: first matrix
    :param wave_offset_port_1: port wave offset. If 0 it is in current wave.
    :param port_input_name_2: second matrix
    :param wave_offset_port_2: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param level:  pyramid level to calculate at
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name_1, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_input_name_2, lvl=level)

    if port_output_name is None:
        port_output_name = 'BITWISE_' + port_input_name_1 + '_AND_' + port_input_name_2

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_1, input_port_2]
    main_func_list = [input_port_1, wave_offset_port_1, input_port_2, wave_offset_port_2, output_port]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Bitwise AND', input_list=input_port_list,
                               wave_offset=[wave_offset_port_1, wave_offset_port_2], level=level)

    d = create_dictionary_element(job_module='processing_multiple_images',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=max(wave_offset_port_1, wave_offset_port_2),
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='and_bitwise_between_2_images',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_matrix_bitwise_or_job(port_input_name_1: str, port_input_name_2: str,
                             wave_offset_port_1: int = 0, wave_offset_port_2: int = 0,
                             port_output_name: str = None,
                             level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Matrix bitwise OR from 2 images.
    Works only on grayscale images.
    :param port_input_name_1: first matrix
    :param wave_offset_port_1: port wave offset. If 0 it is in current wave.
    :param port_input_name_2: second matrix
    :param wave_offset_port_2: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param level: pyramid level to calculate at
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name_1, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_input_name_2, lvl=level)

    if port_output_name is None:
        port_output_name = 'BITWISE_' + port_input_name_1 + '_OR_' + port_input_name_2

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_1, input_port_2]
    main_func_list = [input_port_1, wave_offset_port_1, input_port_2, wave_offset_port_2, output_port]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Bitwise OR', input_list=input_port_list,
                               wave_offset=[wave_offset_port_1, wave_offset_port_2], level=level)

    d = create_dictionary_element(job_module='processing_multiple_images',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=max(wave_offset_port_1, wave_offset_port_2),
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='or_bitwise_between_2_images',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_matrix_bitwise_or_4_job(port_input_name_1: str, port_input_name_2: str, port_input_name_3: str, port_input_name_4: str,
                               wave_offset_port_1: int = 0, wave_offset_port_2: int = 0,
                               wave_offset_port_3: int = 0, wave_offset_port_4: int = 0,
                               port_output_name: str = None,
                               level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Matrix bitwise OR from 4 images.
    Works only on grayscale.
    :param port_input_name_1: first matrix
    :param wave_offset_port_1: port wave offset. If 0 it is in current wave.
    :param port_input_name_2: second matrix
    :param wave_offset_port_2: port wave offset. If 0 it is in current wave.
    :param port_input_name_3: third matrix
    :param wave_offset_port_3: port wave offset. If 0 it is in current wave.
    :param port_input_name_4: fourth matrix
    :param wave_offset_port_4: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param level:  pyramid level to calculate at
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name_1, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_input_name_2, lvl=level)
    input_port_3 = transform_port_name_lvl(name=port_input_name_3, lvl=level)
    input_port_4 = transform_port_name_lvl(name=port_input_name_4, lvl=level)

    if port_output_name is None:
        port_output_name = 'BITWISE_' + port_input_name_1 + '_OR_' + port_input_name_2 + '_OR_' \
                           + port_input_name_3 + '_OR_' + port_input_name_4

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_1, input_port_2, input_port_3, input_port_4]

    main_func_list = [input_port_1, wave_offset_port_1, input_port_2, wave_offset_port_2, input_port_3,
                      wave_offset_port_3, input_port_4, wave_offset_port_4, output_port]

    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Bitwise OR', input_list=input_port_list,
                               wave_offset=[wave_offset_port_1, wave_offset_port_2, wave_offset_port_3, wave_offset_port_4], level=level)

    d = create_dictionary_element(job_module='processing_multiple_images',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=max(wave_offset_port_1, wave_offset_port_2, wave_offset_port_3, wave_offset_port_4),
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='or_bitwise_between_4_images',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_matrix_bitwise_xor_job(port_input_name_1: str, port_input_name_2: str,
                              wave_offset_port_1: int = 0, wave_offset_port_2: int = 0,
                              port_output_name: str = None,
                              level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Matrix XOR from 2 images.
    Works only on grayscale images.
    :param port_input_name_1: first matrix
    :param wave_offset_port_1: port wave offset. If 0 it is in current wave.
    :param port_input_name_2: second matrix
    :param wave_offset_port_2: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param level: pyramid level to calculate at
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name_1, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_input_name_2, lvl=level)

    if port_output_name is None:
        port_output_name = 'BITWISE_' + port_input_name_1 + '_XOR_' + port_input_name_2

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_1, input_port_2]
    main_func_list = [input_port_1, wave_offset_port_1, input_port_2, wave_offset_port_2, output_port]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Bitwise XOR', input_list=input_port_list,
                               wave_offset=[wave_offset_port_1, wave_offset_port_2], level=level)

    d = create_dictionary_element(job_module='processing_multiple_images',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=max(wave_offset_port_1, wave_offset_port_2),
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='xor_bitwise_between_2_images',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_matrix_intersect_job(port_input_name: str, port_input_mask: str,
                            wave_offset_port_1: int = 0, wave_offset_port_2: int = 0,
                            port_output_name: str = None,
                            level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Intersect image with binary mask
    Works only on grayscale images.
    :param port_input_name: image we want to manipulate
    :param wave_offset_port_1: port wave offset. If 0 it is in current wave.
    :param port_input_mask: mask matrix
    :param wave_offset_port_2: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param level:  pyramid level to calculate at
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_input_mask, lvl=level)

    if port_output_name is None:
        port_output_name = 'INTERSECT_' + port_input_name + '_AND_' + port_input_mask

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_1, input_port_2]
    main_func_list = [input_port_1, wave_offset_port_1, input_port_2, wave_offset_port_2, output_port]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Bitwise AND', input_list=input_port_list,
                               wave_offset=[wave_offset_port_1, wave_offset_port_2], level=level)

    d = create_dictionary_element(job_module='processing_multiple_images',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=max(wave_offset_port_1, wave_offset_port_2),
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='intersect_between_2_images',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


############################################################################################################################################
# Pyramid level processing jobs
############################################################################################################################################


def do_pyramid_level_down_job(port_input_name: str, number_of_lvl: int,
                              port_input_lvl: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0,
                              is_rgb: bool = False, wave_offset: int = 0,
                              port_output_name: str = None, verbose: bool = False) -> None:
    """
    Function for configure the image pyramid level calculation job.  The implementation is done using the OpenCV library.
    https://www.researchgate.net/publication/246727904_Pyramid_Methods_in_Image_Processing
    :param port_input_name: name of input port to start from
    :param port_input_lvl: level of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output ports you want
    :param number_of_lvl: how many level do you want to execute
    :param verbose: debug
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: None
    """
    input_name_port = transform_port_name_lvl(name=port_input_name, lvl=port_input_lvl)

    if number_of_lvl > PYRAMID_LEVEL.NUMBER_LVL:
        number_of_lvl = PYRAMID_LEVEL.NUMBER_LVL
        log_to_console('LOWEST DEPTH OF PYRAMID ACCEPTED IS {l}. CHANGED TO {l}'.format(l=PYRAMID_LEVEL.NUMBER_LVL).upper())

    if port_output_name is None:
        port_output_name = port_input_name

    # construct list of output ports
    output_ports = []
    main_f_param = [wave_offset, input_name_port]
    input_port_list = [input_name_port]

    for i in range(1, number_of_lvl + 1, 1):
        # creates the tuple for output ports
        output_ports.append(
            (transform_port_name_lvl(name=port_output_name, lvl=eval('PYRAMID_LEVEL.LEVEL_' + str(i))),
             transform_port_size_lvl(lvl=eval('PYRAMID_LEVEL.LEVEL_' + str(i)), rgb=is_rgb),
             'B',
             True)
        )
        main_f_param.append(output_ports[-1][0])

    if verbose:
        print(output_ports)

    job_name = job_name_create(action='Pyramid Reduce', input_list=input_port_list, wave_offset=[wave_offset], level=port_input_lvl,
                               Levels=str(number_of_lvl))

    # main function needs input port as param
    d = create_dictionary_element(job_module='pyramid_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_down', main_func_param=main_f_param,
                                  output_ports=output_ports)

    jobs_dict.append(d)


def do_pyramid_level_up_job(port_input_name: str, number_of_lvl: int,
                            port_input_lvl: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0,
                            is_rgb: bool = False, wave_offset: int = 0,
                            port_output_name: str = None, verbose: bool = False) -> None:
    """
    Function for configure the image pyramid level calculation job. The implementation is done using the OpenCV library.
    https://www.researchgate.net/publication/246727904_Pyramid_Methods_in_Image_Processing
    :param port_input_name: name of input port to start from
    :param port_input_lvl: level of input port
    :param port_output_name: name of output ports you want
    :param number_of_lvl: how many level do you want to execute
    :param verbose: debug
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: None
    """
    input_name_port = transform_port_name_lvl(name=port_input_name, lvl=port_input_lvl)

    if number_of_lvl > PYRAMID_LEVEL.NUMBER_LVL:
        number_of_lvl = PYRAMID_LEVEL.NUMBER_LVL
        log_to_console('LOWEST DEPTH OF PYRAMID ACCEPTED IS {l}. CHANGED TO {l}'.format(l=PYRAMID_LEVEL.NUMBER_LVL).upper())

    if port_output_name is None:
        port_output_name = 'EXPAND_' + port_input_name

    # construct list of output ports
    output_ports = []
    main_f_param = [wave_offset, input_name_port]
    input_port_list = [input_name_port]

    for i in range(1, number_of_lvl + 1, 1):
        # creates the tuple for output ports
        output_ports.append(
            (transform_port_name_lvl(name=port_output_name, lvl=eval('PYRAMID_LEVEL.LEVEL_' + str(int(str(port_input_lvl)[-1]) - i))),
             transform_port_size_lvl(lvl=eval('PYRAMID_LEVEL.LEVEL_' + str(int(str(port_input_lvl)[-1]) - i)), rgb=is_rgb),
             'B',
             True)
        )
        main_f_param.append(output_ports[-1][0])

    if verbose:
        print(output_ports)

    job_name = job_name_create(action='Pyramid Expand', input_list=input_port_list, wave_offset=[wave_offset], level=port_input_lvl,
                               Levels=str(number_of_lvl))

    # main function needs input port as param
    d = create_dictionary_element(job_module='pyramid_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_up', main_func_param=main_f_param,
                                  output_ports=output_ports)

    jobs_dict.append(d)


############################################################################################################################################
# Image blurring jobs
############################################################################################################################################


def do_gaussian_blur_image_job(port_input_name: str,
                               port_output_name: str = None, kernel_size: int = 0, sigma: float = 0.0,
                               wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False) -> str:
    """
    Function to configure the Gaussian blur on an image job.
    By default the kernel is 3x3 and sigma is auto calculated. The implementation is done using the OpenCV library.
    http://www.ipol.im/pub/art/2013/87/?utm_source=doi
    :param sigma: sigma to use. 0 to auto calculate
    :param kernel_size: kernel of gaussian to use
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'GAUSS_BLUR_K_' + str(kernel_size) + '_S_' + str(sigma).replace('.', '_') + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel_size, sigma, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Gaussian Blur', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               K=str(kernel_size), S=str(sigma))

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  max_wave=wave_offset,
                                  input_ports=input_port_list,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_gaussian_blur_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_median_blur_job(port_input_name: str,
                       kernel_size: int = 3, port_output_name: str = None,
                       is_rgb: bool = False, wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0):
    """
    Function for configure the median filter blur image calculation job.
    By default the kernel is 3x3. The implementation is done using OpenCV-image library.
    https://www.uio.no/studier/emner/matnat/ifi/INF2310/v12/undervisningsmateriale/artikler/Huang-etal-median.pdf
    :param kernel_size: kernel of gaussian to use
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'MEDIAN_BLUR_K_' + str(kernel_size) + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel_size, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Median Blur', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               K=str(kernel_size))

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_median_blur_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_mean_blur_job(port_input_name: str,
                     kernel_size: int = 3, port_output_name: str = None,
                     is_rgb: bool = False, wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    The idea of mean filtering is simply to replace each pixel value in an image with the mean (`average') 
    value of its neighbors, including itself. This has the effect of eliminating pixel values which are unrepresentative
    of their surroundings. The implementation is done using OpenCV-image library.
    By default the kernel is 3x3.
    http://homepages.inf.ed.ac.uk/rbf/BOOKS/VERNON/Chap001.pdf
    :param kernel_size: kernel of gaussian to use
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'MEAN_BLUR_K_' + str(kernel_size) + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel_size, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Mean Blur', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               K=str(kernel_size))

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_mean_blur_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_conservative_filter_job(port_input_name: str,
                               kernel_size: int = 3, port_output_name: str = None,
                               wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Conservative smoothing is a noise reduction technique that derives its name from the fact that it employs a simple, 
    fast filtering algorithm that sacrifices noise suppression power in order to preserve the high spatial frequency detail
    (e.g. sharp edges) in an image.
    Works only on greyscale images
    By default the kernel is 3x3 and sigma is 1
    http://homepages.inf.ed.ac.uk/rbf/BOOKS/VERNON/Chap001.pdf
    :param kernel_size: kernel of gaussian to use
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'CONSERVATIVE_K_' + str(kernel_size) + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel_size, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Conservative filter', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               K=str(kernel_size))

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_conservative_filter_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_crimmins_job(port_input_name: str,
                    port_output_name: str = None,
                    wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Crimmins Speckle Removal reduces speckle from an image using the Crimmins complementary hulling algorithm.
    The algorithm has been specifically designed to reduce the intensity of salt and pepper noise in an image.
    Increased iterations of the algorithm yield increased levels of noise removal, but also introduce a significant amount of
    blurring of high frequency details.
    Works only on grayscale.
    https://www.osapublishing.org/ao/abstract.cfm?uri=ao-24-10-1438
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'CRIMMINS_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Crimmins Filter', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_crimmins_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_unsharp_filter_job(port_input_name: str,
                          radius: int = 2, percent: int = 150, port_output_name: str = None,
                          wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    The unsharp filter is a simple sharpening operator which derives its name from the fact that it enhances edges
    (and other high frequency components in an image) via a procedure which subtracts an unsharp, or smoothed,
    version of an image from the original image. The unsharp filtering technique is commonly used in the photographic
    and printing industries for crisping edges. The implementation is done using PIL-image library.
    By default the radius is 2 and percent is 150.
    https://pdfs.semanticscholar.org/a9ea/aecf23f3a4b7822e4bcca924e02cd5b4dc4e.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param radius: radius of filter
    :param percent: percent of dark to add
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    return port_output_name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'UNSHARP_FILER_R_' + str(radius) + '_P_' + str(percent) + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, radius, percent, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Unsharp filter', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               R=str(radius), P=str(percent))

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_unsharp_filter_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_isef_filter_job(port_input_name: str,
                       smoothing_factor: float = 0.9, port_output_name: str = None,
                       wave_offset: int = 0, is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    The function that minimizes CN is the optimal smoothing filter for an edge detector.
    The optimal filter function they came up with is the Infinite Symmetric Exponential Filter (ISEF)
    https://ieeexplore.ieee.org/abstract/document/118199
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param smoothing_factor: factor to smooth 0-1
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'ISEF_FILER_B_' + str(smoothing_factor) + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, smoothing_factor, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'h', True)]

    job_name = job_name_create(action='ISEF', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               B=str(smoothing_factor))

    d = create_dictionary_element(job_module='edge_shen_castan',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_isef', init_func_param=None,
                                  main_func_name='main_isef_smoothing',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_bilateral_filter_job(port_input_name: str,
                            distance: int = 9, sigma_colors: int = 75, sigma_space: int = 75, port_output_name: str = None,
                            is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    A bilateral filter is used for smoothing images and reducing noise, while preserving edges.
    The implementation is done using OpenCV-image library.
    http://www.cse.ucsc.edu/~manduchi/Papers/ICCV98.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param distance: kernel of gaussian to use
    :param sigma_colors: The greater the value, the colors farther to each other will start to get mixed.
    :param sigma_space: he greater its value, the more further pixels will mix together.
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'BILATERAL_D_' + str(distance) + '_SC_' + str(sigma_colors) + '_SS_' + str(sigma_space) \
                           + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, distance, sigma_colors, sigma_space, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Bilateral filter', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               D=str(distance), SC=str(sigma_colors), SS=str(sigma_space))

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_bilateral_blur_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_guided_filter_job(port_input_name: str,
                         radius: int = 8, regularization: float = 0.4, port_output_name: str = None,
                         level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    Derived from a local linear model, the guided filter computes the filtering output by considering the content of a guidance image,
    which can be the input image itself or another different image. The guided filter is also a more generic concept beyond smoothing:
    it can transfer the structures of the guidance image to the filtering output, enabling new filtering applications like
    dehazing and guided feathering. The implementation is done using OpenCV-image library.
    This implementation will consider the image == guide image
    http://kaiminghe.com/eccv10/
    http://kaiminghe.com/publications/pami12guidedfilter.pdf
    https://arxiv.org/abs/1505.00996
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param radius: radius to use
    :param regularization: regularization term of Guided Filter. \f${eps}^2\f$
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'GUIDED_R_' + str(radius) + '_EPS_' + str(regularization).replace('.', '_') \
                           + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, radius, regularization, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Guided filter', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               R=str(radius), EPS=str(regularization))

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_guided_filter_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_l0_gradient_minimization_filter_job(port_input_name: str,
                                           lambda_value: float = None, kappa_value: float = None, port_output_name: str = None,
                                           level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0,
                                           is_rgb: bool = False) -> str:
    """
    Global image smoothing via L0 gradient minimization.The seemingly contradiction effect is achieved in an unconventional optimization
    framework making use of L0 gradient minimization, which can globally control how many non-zero gradients are resulted to approximate
    prominent structures in a structure-sparsity-management manner. The implementation is done using OpenCV-image library.
    #https://sites.fas.harvard.edu/~cs278/papers/l0.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param lambda_value: lambda parameter defining the smooth term weight.
    :param kappa_value: kappa parameter defining the increasing factor of the weight of the gradient data term.
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'L0_GRADIENT_MINIMIZATION_L_' + str(lambda_value).replace('.', '_') + '_K_' + str(kappa_value).replace('.', '_') \
                           + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, lambda_value, kappa_value, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='L0 gradient minimization filter', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               L=str(lambda_value), K=str(kappa_value))

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_l0_smoothing_filter_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_anisotropic_diffusion_filter_job(port_input_name: str,
                                        alpha: float = 0.5, kappa: float = 0.2, niter: int = 1, port_output_name: str = None,
                                        is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    The function applies Perona-Malik anisotropic diffusion to an image. High-frequency noise is present in magnetic resonance images and
    it is usually removed by a filtering process. The anisotropic diffusion filter (ADF) was proposed to adaptively remove the noise,
    maintaining the image edges. The implementation is done using OpenCV-image library.
    https://www2.eecs.berkeley.edu/Pubs/TechRpts/1988/CSD-88-483.pdf
    https://ieeexplore.ieee.org/document/56205
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param alpha: The amount of time to step forward by on each iteration (normally, it's between 0 and 1).
    :param kappa: sensitivity to the edges
    :param niter: The number of iterations
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'ANISOTROPIC_DIFFUSION_A_' + str(alpha).replace('.', '_') + '_K_' + str(kappa).replace('.', '_') + \
                           '_N_' + str(niter) + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, alpha, kappa, niter, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Anisotropic Diffusion filter', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               Alpha=str(alpha), Kappa=str(kappa), Niter=str(niter))

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_anisotropic_diffusion_filter_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_motion_blur_filter_job(port_input_name: str,
                              kernel_size: int, angle: float, port_output_name: str = None,
                              is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Motion blur is the apparent streaking of moving objects in a photograph or a sequence of frames, such as a film or animation. It results when the image 
    being recorded changes during the recording of a single exposure, due to rapid movement or long exposure.
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel_size: kernel size of motion blur filter
    :param angle: angle of motion blur filter
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'MOTION_BLUR_K_' + str(kernel_size).replace('.', '_') + '_ANGLE_' + str(angle).replace('.',
                                                                                                                  '_') + '_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, kernel_size, angle, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Motion blur filter', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               Kernel_size=kernel_size, Angle=angle)

    d = create_dictionary_element(job_module='blur_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_motion_blur_filter_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


###########################################################################################################################################
# Image morphology jobs
###########################################################################################################################################


# noinspection PyTypeChecker
def do_image_morphological_erosion_job(port_input_name: str,
                                       kernel_to_use: list = MORPH_CONFIG.KERNEL_RECTANGULAR, kernel_size: int = 3,
                                       input_iteration: int = 1, port_output_name: str = None,
                                       level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Erosion is one of the two basic operators in the area of mathematical morphology, the other being dilation. It is typically applied
    to binary images, but there are versions that work on grayscale images. The basic effect of the operator on a binary image is to
    erode away the boundaries of regions of foreground pixels. Thus areas of foreground pixels shrink in size, and holes within those
    areas become larger. The implementation is done using OpenCV-image library.
    https://www.sciencedirect.com/science/article/abs/pii/0734189X86900022
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of desired output port
    :param level:  pyramid level to calculate at
    :param kernel_to_use: custom kernel to use. Please use a list of lists to describe the array.
                              eg:  [[1,2,1], [2,2,2], [3,2,3]]
                         Or a kernel existing in the config_main module in MORPH_CONFIG
    :param kernel_size: Kernel size
    :param input_iteration: number of iterations
    :return: output image port name
    """
    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

    # check kernel passed
    if isinstance(kernel_to_use, str) and kernel_to_use in MORPH_CONFIG.__dict__.values():
        kernel = (cv2.getStructuringElement(shape=eval(kernel_to_use), ksize=(kernel_size, kernel_size))).tolist()
        kernel_name = kernel_to_use.split('_')[-1]
    elif isinstance(kernel_to_use, list):
        kernel = kernel_to_use
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel_name = 'CUSTOM_' + str(custom_kernels_used.index(kernel_to_use))
        kernel_size = len(kernel)
    else:
        log_setup_info_to_console("MORPHOLOGICAL EROSION JOB DIDN'T RECEIVE CORRECT KERNEL")
        return

    if port_output_name is None:
        port_output_name = 'MORPH_ERODED_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(input_iteration) + '_' + port_input_name

    output_port_img = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port]
    main_func_list = [input_port, wave_offset, kernel.__str__(), input_iteration, output_port_img]
    output_port_list = [(output_port_img, output_port_size, 'B', True)]

    job_name = job_name_create(action='Morph Erosion', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               Kernel=str(kernel_name), Kernel_size=str(kernel_size), Iteration=str(input_iteration))

    d = create_dictionary_element(job_module='morphological_operations',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_erosion',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


# noinspection PyTypeChecker
def do_image_morphological_dilation_job(port_input_name: str,
                                        kernel_to_use: list = MORPH_CONFIG.KERNEL_RECTANGULAR, kernel_size: int = 3,
                                        input_iteration: int = 1, port_output_name: str = None,
                                        level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    The assigned structuring element is used for probing and expanding the shapes contained in the input image. In Specific, it acts like
    local maximum filter. Dilation has the opposite effect to erosion. It adds a layer of pixels to both the inner and outer boundaries of
    regions. That is, the value of the output pixel is the maximum value of all pixels in the neighborhood. In a binary image, a pixel is
    set to 1 if any of the neighboring pixels have the value 1. Morphological dilation makes objects more visible and fills in small holes
    in objects. The implementation is done using OpenCV-image library.
    https://www.sciencedirect.com/science/article/abs/pii/0734189X86900022
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of desired output port
    :param level:  pyramid level to calculate at
    :param kernel_to_use: custom kernel to use. Please use a list of lists to describe the array.
                              eg:  [[1,2,1], [2,2,2], [3,2,3]]
                         Or a kernel existing in the config_main module in MORPH_CONFIG
    :param kernel_size: Kernel size
    :param input_iteration: number of iterations
    :return: output image port name
    """
    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

    # check kernel passed
    if isinstance(kernel_to_use, str) and kernel_to_use in MORPH_CONFIG.__dict__.values():
        kernel = (cv2.getStructuringElement(shape=eval(kernel_to_use), ksize=(kernel_size, kernel_size))).tolist()
        kernel_name = kernel_to_use.split('_')[-1]
    elif isinstance(kernel_to_use, list):
        kernel = kernel_to_use
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel_name = 'CUSTOM_' + str(custom_kernels_used.index(kernel_to_use))
        kernel_size = len(kernel)
    else:
        log_setup_info_to_console("MORPHOLOGICAL DILATION JOB DIDN'T RECEIVE CORRECT KERNEL")
        return

    if port_output_name is None:
        port_output_name = 'MORPH_DILATED_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(input_iteration) + '_' + port_input_name

    output_port_img = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port]
    main_func_list = [input_port, wave_offset, kernel.__str__(), input_iteration, output_port_img]
    output_port_list = [(output_port_img, output_port_size, 'B', True)]

    job_name = job_name_create(action='Morph Dilation', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               Kernel=str(kernel_name), Kernel_size=str(kernel_size), Iteration=str(input_iteration))

    d = create_dictionary_element(job_module='morphological_operations',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_dilation',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_image_morphological_cv2_job(port_input_name: str, port_output_name: str, operation_to_use: str,
                                   input_structural_element: str = 'cv2.MORPH_CROSS', input_structural_kernel: int = 3,
                                   input_iteration: int = 1, use_custom_kernel: np.array = None,
                                   level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> None:
    """
    The assigned structuring element is used for probing and expanding the shapes contained in the input image. In Specific, it acts like
    local maximum filter. Dilation has the opposite effect to erosion. It adds a layer of pixels to both the inner and outer boundaries
    of regions. That is, the value of the output pixel is the maximum value of all pixels in the neighborhood. In a binary image, a pixel
    is set to 1 if any of the neighboring pixels have the value 1. Morphological dilation makes objects more visible and fills in small
    holes in objects. The implementation is done using OpenCV-image library.
    https://www.sciencedirect.com/science/article/abs/pii/0734189X86900022
    :param port_input_name: name of input port
    :param port_output_name: name of desired output port
    :param level:  pyramid level to calculate at
    :param use_custom_kernel: custom kernel to use. Please use a list of lists to describe the array.
                              eg:  [[1,2,1], [2,2,2], [3,2,3]]
    :param input_structural_element: The function constructs and returns the structuring element that can be
                                    further passed to #erode, #dilate or #morphologyEx.
                                    MORPH_RECT = 0
                                    MORPH_CROSS = 1
                                    MORPH_ELLIPSE = 2
    :param operation_to_use: what operations to use
                                    MORPH_ERODE = 0
                                    MORPH_DILATE = 1
                                    MORPH_OPEN = 2
                                    MORPH_CLOSE = 3
                                    MORPH_GRADIENT = 4
                                    MORPH_TOPHAT = 5
                                    MORPH_BLACKHAT = 6
                                    MORPH_HITMISS = 7
    :param input_structural_kernel: Kernel size
    :param input_iteration: number of iterations
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: None
    """

    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)
    output_port_img = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port]
    kernel = str(use_custom_kernel)
    input_structural_kernel = len(use_custom_kernel) if use_custom_kernel is not None else input_structural_kernel

    main_func_list = [input_port, wave_offset, operation_to_use, input_structural_element, input_structural_kernel, input_iteration, kernel,
                      output_port_img]

    output_port_list = [(output_port_img, output_port_size, 'B', True)]

    job_name = job_name_create(action='Morph operation', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               Op=str(operation_to_use.split('_')[-1]), Iteration=str(input_iteration),
                               Kernel=str(input_structural_element),
                               Kernel_size=str(input_structural_kernel))

    d = create_dictionary_element(job_module='morphological_operations',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_morphology_ex',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


# noinspection PyTypeChecker
def do_image_morphological_open_job(port_input_name: str,
                                    kernel_to_use: list = MORPH_CONFIG.KERNEL_RECTANGULAR, kernel_size: int = 3,
                                    input_iteration: int = 1, port_output_name: str = None,
                                    level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0,
                                    do_fast: bool = True) -> str:
    """
    Morphological opening on an image is defined as erosion of I by B, followed by dilation of the eroded image with B. By using opening
    operation, external noise is removed which is present in the background region and object keeps as it is original. The implementation
    is done using OpenCV-image library.
    https://www.researchgate.net/profile/Robert_Haralick/publication/240038641_Image_Algebra_Using_Mathematical_Morphology/links/571e358408aed056fa2268b6/Image-Algebra-Using-Mathematical-Morphology.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of desired output port
    :param level:  pyramid level to calculate at
    :param kernel_to_use: custom kernel to use. Please use a list of lists to describe the array.
                              eg:  [[1,2,1], [2,2,2], [3,2,3]]
                         Or a kernel existing in the config_main module in MORPH_CONFIG
    :param kernel_size: Kernel size
    :param input_iteration: number of iterations
    :param do_fast: use directly cv2 implementation
    :return: output image port name
    """
    # check kernel passed
    if isinstance(kernel_to_use, str) and kernel_to_use in MORPH_CONFIG.__dict__.values():
        kernel = (cv2.getStructuringElement(shape=eval(kernel_to_use), ksize=(kernel_size, kernel_size))).tolist()
        kernel_name = kernel_to_use.split('_')[-1]
    elif isinstance(kernel_to_use, list):
        kernel = kernel_to_use
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel_name = 'CUSTOM_' + str(custom_kernels_used.index(kernel_to_use))
        kernel_size = len(kernel)
    else:
        log_setup_info_to_console("MORPHOLOGICAL OPEN JOB DIDN'T RECEIVE CORRECT KERNEL")
        return

    if port_output_name is None:
        port_output_name = 'MORPH_OPEN_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(input_iteration) + '_' + port_input_name
        if do_fast:
            port_output_name += '_FAST'

    if do_fast:
        do_image_morphological_cv2_job(port_input_name=port_input_name,
                                       port_output_name=port_output_name,
                                       level=level, wave_offset=wave_offset,
                                       operation_to_use='cv2.MORPH_OPEN',
                                       input_structural_element=kernel_to_use,
                                       input_structural_kernel=kernel_size,
                                       input_iteration=input_iteration,
                                       use_custom_kernel=kernel)
    else:
        erode_output_name = 'MORPH_ERODED_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(input_iteration) + '_' + port_input_name

        do_image_morphological_erosion_job(port_input_name=port_input_name,
                                           wave_offset=wave_offset,
                                           port_output_name=erode_output_name,
                                           kernel_to_use=kernel_to_use,
                                           kernel_size=kernel_size,
                                           input_iteration=input_iteration,
                                           level=level)

        do_image_morphological_dilation_job(port_input_name=erode_output_name,
                                            wave_offset=0,
                                            port_output_name=port_output_name,
                                            kernel_to_use=kernel_to_use,
                                            kernel_size=kernel_size,
                                            input_iteration=input_iteration,
                                            level=level)

    return port_output_name


# noinspection PyTypeChecker
def do_image_morphological_close_job(port_input_name: str,
                                     port_output_name: str = None, kernel_to_use: list = MORPH_CONFIG.KERNEL_RECTANGULAR,
                                     kernel_size: int = 3, input_iteration: int = 1,
                                     level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0,
                                     do_fast: bool = True) -> str:
    """
    Closing is opening performed in reverse. It is defined simply as a dilation followed by an erosion using the same structuring element
    for both operations. See the sections on erosion and dilation for details of the individual steps. The closing operator therefore
    requires two inputs: an image to be closed and a structuring element. The implementation is done using OpenCV-image library.
    https://www.researchgate.net/profile/Robert_Haralick/publication/240038641_Image_Algebra_Using_Mathematical_Morphology/links/571e358408aed056fa2268b6/Image-Algebra-Using-Mathematical-Morphology.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of desired output port
    :param level:  pyramid level to calculate at
    :param kernel_to_use: custom kernel to use. Please use a list of lists to describe the array.
                              eg:  [[1,2,1], [2,2,2], [3,2,3]]
                         Or a kernel existing in the config_main module in MORPH_CONFIG
    :param kernel_size: Kernel size
    :param input_iteration: number of iterations
    :param do_fast: if we want to use the CV2 job
    return port_output_name
    """
    # check kernel passed
    if isinstance(kernel_to_use, str) and kernel_to_use in MORPH_CONFIG.__dict__.values():
        kernel = (cv2.getStructuringElement(shape=eval(kernel_to_use), ksize=(kernel_size, kernel_size))).tolist()
        kernel_name = kernel_to_use.split('_')[-1]
    elif isinstance(kernel_to_use, list):
        kernel = kernel_to_use
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel_name = 'CUSTOM_' + str(custom_kernels_used.index(kernel_to_use))
        kernel_size = len(kernel)
    else:
        log_setup_info_to_console("MORPHOLOGICAL CLOSE JOB DIDN'T RECEIVE CORRECT KERNEL")
        return

    if port_output_name is None:
        port_output_name = 'MORPH_CLOSE_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(input_iteration) + '_' + port_input_name
        if do_fast:
            port_output_name += '_FAST'

    if do_fast:
        do_image_morphological_cv2_job(port_input_name=port_input_name,
                                       port_output_name=port_output_name,
                                       level=level, wave_offset=wave_offset,
                                       operation_to_use='cv2.MORPH_CLOSE',
                                       input_structural_element=kernel_to_use,
                                       input_structural_kernel=kernel_size,
                                       input_iteration=input_iteration,
                                       use_custom_kernel=kernel)
    else:
        output_dilation_name = 'MORPH_DILATED_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(
            input_iteration) + '_' + port_input_name

        do_image_morphological_dilation_job(port_input_name=port_input_name,
                                            wave_offset=wave_offset,
                                            port_output_name=output_dilation_name,
                                            kernel_to_use=kernel_to_use,
                                            kernel_size=kernel_size,
                                            input_iteration=input_iteration,
                                            level=level)

        do_image_morphological_erosion_job(port_input_name=output_dilation_name,
                                           wave_offset=0,
                                           port_output_name=port_output_name,
                                           kernel_to_use=kernel_to_use,
                                           kernel_size=kernel_size,
                                           input_iteration=input_iteration,
                                           level=level)

    return port_output_name


# noinspection PyTypeChecker
def do_image_morphological_edge_gradient_job(port_input_name: str,
                                             port_output_name: str = None, kernel_to_use: list = MORPH_CONFIG.KERNEL_RECTANGULAR,
                                             kernel_size: int = 3, input_iteration: int = 1,
                                             level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0,
                                             do_fast=True) -> str:
    """
    The difference between the dilation and the erosion of the image. This is used to find boundaries or edges in an image. It
    recommended to apply some filtering before calculating the gradient because it is very sensitive to noise. The implementation is done
    using OpenCV-image library.
    https://www.researchgate.net/profile/Robert_Haralick/publication/240038641_Image_Algebra_Using_Mathematical_Morphology/links/571e358408aed056fa2268b6/Image-Algebra-Using-Mathematical-Morphology.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of desired output port
    :param level:  pyramid level to calculate at
    :param kernel_to_use: custom kernel to use. Please use a list of lists to describe the array.
                              eg:  [[1,2,1], [2,2,2], [3,2,3]]
                         Or a kernel existing in the config_main module in MORPH_CONFIG
    :param kernel_size: Kernel size
    :param input_iteration: number of iterations
    :param do_fast: if we want to use the CV2 job
    :return: output image port name
    """
    # check kernel passed
    if isinstance(kernel_to_use, str) and kernel_to_use in MORPH_CONFIG.__dict__.values():
        kernel = (cv2.getStructuringElement(shape=eval(kernel_to_use), ksize=(kernel_size, kernel_size))).tolist()
        kernel_name = kernel_to_use.split('_')[-1]
    elif isinstance(kernel_to_use, list):
        kernel = kernel_to_use
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel_name = 'CUSTOM_' + str(custom_kernels_used.index(kernel_to_use))
        kernel_size = len(kernel)
    else:
        log_setup_info_to_console("MORPHOLOGICAL EDGE GRADIENT JOB DIDN'T RECEIVE CORRECT KERNEL")
        return

    if port_output_name is None:
        port_output_name = 'MORPH_EDGE_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(input_iteration) + '_' + port_input_name
        if do_fast:
            port_output_name += '_FAST'

    if do_fast:
        do_image_morphological_cv2_job(port_input_name=port_input_name,
                                       port_output_name=port_output_name,
                                       level=level, wave_offset=wave_offset,
                                       operation_to_use='cv2.MORPH_GRADIENT',
                                       input_structural_element=kernel_to_use,
                                       input_structural_kernel=kernel_size,
                                       input_iteration=input_iteration,
                                       use_custom_kernel=kernel)
    else:
        erode_output_name = 'MORPH_ERODED_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(input_iteration) + '_' + port_input_name
        output_dilation_name = 'MORPH_DILATED_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(
            input_iteration) + '_' + port_input_name

        do_image_morphological_dilation_job(port_input_name=port_input_name,
                                            wave_offset=wave_offset,
                                            port_output_name=output_dilation_name,
                                            kernel_to_use=kernel_to_use,
                                            kernel_size=kernel_size,
                                            input_iteration=input_iteration,
                                            level=level)

        do_image_morphological_erosion_job(port_input_name=port_input_name,
                                           wave_offset=wave_offset,
                                           port_output_name=erode_output_name,
                                           kernel_to_use=kernel_to_use,
                                           kernel_size=kernel_size,
                                           input_iteration=input_iteration,
                                           level=level)

        do_matrix_difference_job(port_input_name_1=output_dilation_name,
                                 port_input_name_2=erode_output_name,
                                 port_output_name=port_output_name,
                                 level=level)

    return port_output_name


# noinspection PyTypeChecker
def do_image_morphological_top_hat_job(port_input_name: str,
                                       port_output_name: str = None, kernel_to_use: list = MORPH_CONFIG.KERNEL_RECTANGULAR,
                                       kernel_size: int = 3, input_iteration: int = 1,
                                       level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0,
                                       do_fast: bool = True) -> str:
    """
    The difference of the source or input image and the opening of the source or input image. It highlights the narrow pathways
    between different regions. The implementation is done using OpenCV-image library.
    https://www.researchgate.net/profile/Robert_Haralick/publication/240038641_Image_Algebra_Using_Mathematical_Morphology/links/571e358408aed056fa2268b6/Image-Algebra-Using-Mathematical-Morphology.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of desired output port
    :param level:  pyramid level to calculate at
    :param kernel_to_use: custom kernel to use. Please use a list of lists to describe the array.
                              eg:  [[1,2,1], [2,2,2], [3,2,3]]
                         Or a kernel existing in the config_main module in MORPH_CONFIG
    :param kernel_size: Kernel size
    :param input_iteration: number of iterations
    :param do_fast: if we want to use the CV2 morphological job
    :return: output image port name
    """
    # check kernel passed
    if isinstance(kernel_to_use, str) and kernel_to_use in MORPH_CONFIG.__dict__.values():
        kernel = (cv2.getStructuringElement(shape=eval(kernel_to_use), ksize=(kernel_size, kernel_size))).tolist()
        kernel_name = kernel_to_use.split('_')[-1]
    elif isinstance(kernel_to_use, list):
        kernel = kernel_to_use
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel_name = 'CUSTOM_' + str(custom_kernels_used.index(kernel_to_use))
        kernel_size = len(kernel)
    else:
        log_setup_info_to_console("MORPHOLOGICAL TOP HAT JOB DIDN'T RECEIVE CORRECT KERNEL")
        return

    if port_output_name is None:
        port_output_name = 'MORPH_TOP_HAT_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(input_iteration) + '_' + port_input_name
        if do_fast:
            port_output_name += '_FAST'

    if do_fast:
        do_image_morphological_cv2_job(port_input_name=port_input_name,
                                       port_output_name=port_output_name,
                                       level=level, wave_offset=wave_offset,
                                       operation_to_use='cv2.MORPH_TOPHAT',
                                       input_structural_element=kernel_to_use,
                                       input_structural_kernel=kernel_size,
                                       input_iteration=input_iteration,
                                       use_custom_kernel=kernel)
    else:
        output_open_name = 'MORPH_OPEN_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(input_iteration) + '_' + port_input_name

        do_image_morphological_open_job(port_input_name=port_input_name,
                                        wave_offset=wave_offset,
                                        port_output_name=output_open_name,
                                        kernel_to_use=kernel_to_use,
                                        kernel_size=kernel_size,
                                        input_iteration=input_iteration,
                                        level=level)

        do_matrix_difference_job(port_input_name_1=port_input_name,
                                 wave_offset_port_1=0,
                                 port_input_name_2=output_open_name,
                                 wave_offset_port_2=0 + wave_offset,
                                 port_output_name=port_output_name,
                                 level=level)

    return port_output_name


# noinspection PyTypeChecker
def do_image_morphological_black_hat_job(port_input_name: str,
                                         port_output_name: str = None, kernel_to_use: list = MORPH_CONFIG.KERNEL_RECTANGULAR,
                                         kernel_size: int = 3, input_iteration: int = 1,
                                         level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0,
                                         do_fast: bool = False) -> str:
    """
    The difference between the closing of an image and the input image itself. This highlights the narrow black regions in the image.
    The implementation is done using OpenCV-image library.
    https://www.researchgate.net/profile/Robert_Haralick/publication/240038641_Image_Algebra_Using_Mathematical_Morphology/links/571e358408aed056fa2268b6/Image-Algebra-Using-Mathematical-Morphology.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of desired output port
    :param level:  pyramid level to calculate at
    :param kernel_to_use: custom kernel to use. Please use a list of lists to describe the array.
                              eg:  [[1,2,1], [2,2,2], [3,2,3]]
                         Or a kernel existing in the config_main module in MORPH_CONFIG
    :param kernel_size: Kernel size
    :param input_iteration: number of iterations
    :param do_fast: use directly cv2 operations
    :return: output image port name
    """
    # check kernel passed
    if isinstance(kernel_to_use, str) and kernel_to_use in MORPH_CONFIG.__dict__.values():
        kernel = (cv2.getStructuringElement(shape=eval(kernel_to_use), ksize=(kernel_size, kernel_size))).tolist()
        kernel_name = kernel_to_use.split('_')[-1]
    elif isinstance(kernel_to_use, list):
        kernel = kernel_to_use
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel_name = 'CUSTOM_' + str(custom_kernels_used.index(kernel_to_use))
        kernel_size = len(kernel)
    else:
        log_setup_info_to_console("MORPHOLOGICAL BLACK HAT JOB DIDN'T RECEIVE CORRECT KERNEL")
        return

    if port_output_name is None:
        port_output_name = 'MORPH_BLACK_HAT_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(
            input_iteration) + '_' + port_input_name
        if do_fast:
            port_output_name += '_FAST'

    if do_fast:
        do_image_morphological_cv2_job(port_input_name=port_input_name,
                                       port_output_name=port_output_name,
                                       level=level, wave_offset=wave_offset,
                                       operation_to_use='cv2.MORPH_BLACKHAT',
                                       input_structural_element=kernel_to_use,
                                       input_structural_kernel=kernel_size,
                                       input_iteration=input_iteration,
                                       use_custom_kernel=kernel)
    else:
        output_close_name = 'MORPH_CLOSE_K_' + str(kernel_size) + '_' + kernel_name + '_IT_' + str(input_iteration) + '_' + port_input_name

        do_image_morphological_close_job(port_input_name=port_input_name,
                                         wave_offset=wave_offset,
                                         port_output_name=output_close_name,
                                         kernel_to_use=kernel_to_use,
                                         kernel_size=kernel_size,
                                         input_iteration=input_iteration,
                                         level=level)

        do_matrix_difference_job(port_input_name_1=output_close_name,
                                 wave_offset_port_1=0,
                                 port_input_name_2=port_input_name,
                                 wave_offset_port_2=0 + wave_offset,
                                 port_output_name=port_output_name,
                                 level=level)

    return port_output_name


# noinspection PyTypeChecker
def do_morphological_hit_and_miss_transformation_job(port_input_name: str,
                                                     port_output_name: str = None,
                                                     use_custom_kernel: np.array = MORPH_CONFIG.KERNEL_HIT_MISS,
                                                     level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    The Hit-or-Miss transformation is useful to find patterns in binary images. In particular, it finds those pixels whose neighbourhood
    matches the shape of a first structuring element B1 while not matching the shape of a second structuring element B2 at the same time.
    The implementation is done using OpenCV-image library.
    https://www.researchgate.net/profile/Robert_Haralick/publication/240038641_Image_Algebra_Using_Mathematical_Morphology/links/571e358408aed056fa2268b6/Image-Algebra-Using-Mathematical-Morphology.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of desired output port
    :param level:  pyramid level to calculate at
    :param use_custom_kernel: custom kernel to use. Please use a list of lists to describe the array.
                              eg:  [[1,2,1], [2,2,2], [3,2,3]]
                              Or a kernel existing in the config_main module in MORPH_CONFIG
    :return: output image port name
    """
    # check kernel passed
    if isinstance(use_custom_kernel, str) and use_custom_kernel in MORPH_CONFIG.__dict__.values():
        kernel_name = use_custom_kernel.split('_')[-1]
    elif isinstance(use_custom_kernel, list):
        kernel = use_custom_kernel
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel_name = 'CUSTOM_' + str(custom_kernels_used.index(use_custom_kernel))
        kernel_size = len(kernel)
    else:
        log_setup_info_to_console("MORPHOLOGICAL HIT MISS JOB DIDN'T RECEIVE CORRECT KERNEL")
        return

    if port_output_name is None:
        # noinspection PyUnboundLocalVariable
        port_output_name = 'MORPH_HIT_MISS_K_' + str(kernel_size) + '_' + kernel_name + '_' + port_input_name

    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)
    output_port_img = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port]
    kernel_b1 = ((np.array(use_custom_kernel) == 1) * 1).tolist()
    kernel_b2 = ((np.array(use_custom_kernel) == -1) * 1).tolist()
    main_func_list = [input_port, wave_offset, kernel_b1.__str__(), kernel_b2.__str__(), output_port_img]
    output_port_list = [(output_port_img, output_port_size, 'B', True)]

    job_name = job_name_create(action='Morph Hit Miss', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               Kernel=str(kernel_name), Kernel_size=str(kernel_size))

    d = create_dictionary_element(job_module='morphological_operations',
                                  job_name=job_name,
                                  max_wave=wave_offset,
                                  input_ports=input_port_list,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_hit_miss',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


# noinspection PyTypeChecker,PyTypeChecker
def do_morphological_thinning_job(port_input_name: str,
                                  use_custom_kernel_1: np.array = MORPH_CONFIG.KERNEL_THINNING_1,
                                  use_custom_kernel_2: np.array = MORPH_CONFIG.KERNEL_THINNING_2,
                                  port_output_name: str = None, input_iteration: int = 1,
                                  level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    This jobs adds one thinning job for each iteration. Like other morphological operators, the behavior of the thinning operation is
    determined by a structuring element. The binary structuring elements used for thinning are of the extended type described under the
    hit-and-miss transform. The implementation is done using OpenCV-image library.
    https://www.researchgate.net/profile/Robert_Haralick/publication/240038641_Image_Algebra_Using_Mathematical_Morphology/links/571e358408aed056fa2268b6/Image-Algebra-Using-Mathematical-Morphology.pdf
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of desired output port
    :param level:  pyramid level to calculate at
    :param use_custom_kernel_1:
    :param use_custom_kernel_2: custom kernel to use. Please use a list of lists to describe the array.
                                eg:  [[1,2,1], [2,2,2], [3,2,3]]
                                Or a kernel existing in the config_main module in MORPH_CONFIG
    :param input_iteration: number of iterations
    :return: output image port name
    """
    # check kernel passed
    if isinstance(use_custom_kernel_1, list):
        kernel_1 = use_custom_kernel_1
        if kernel_1 not in custom_kernels_used:
            custom_kernels_used.append(kernel_1)
        kernel_1_name = 'CUSTOM_' + str(custom_kernels_used.index(kernel_1))
        kernel_1_size = len(kernel_1)
    else:
        log_setup_info_to_console("MORPHOLOGICAL THINNING JOB DIDN'T RECEIVE CORRECT KERNEL")
        return

    # check kernel passed
    if isinstance(use_custom_kernel_2, list):
        kernel_2 = use_custom_kernel_2
        if kernel_2 not in custom_kernels_used:
            custom_kernels_used.append(kernel_2)
        kernel_2_name = 'CUSTOM_' + str(custom_kernels_used.index(kernel_2))
        kernel_2_size = len(kernel_2)
    else:
        log_setup_info_to_console("MORPHOLOGICAL THINNING JOB DIDN'T RECEIVE CORRECT KERNEL")
        return

    if port_output_name is None:
        port_output_name = 'MORPH_THINNING_K_' + str(kernel_1_size) + '_' + kernel_1_name + '_K_' + str(kernel_2_size) \
                           + '_' + kernel_2_name + '_' + port_input_name

    use_custom_kernel = np.array(use_custom_kernel_1)
    kernel_b1 = ((use_custom_kernel == 1) * 1).tolist()
    kernel_b2 = ((use_custom_kernel == -1) * 1).tolist()

    use_custom_kernel = np.array(use_custom_kernel_2)
    kernel_b3 = ((use_custom_kernel == 1) * 1).tolist()
    kernel_b4 = ((use_custom_kernel == -1) * 1).tolist()

    output = port_output_name

    for i in range(input_iteration):
        input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

        if input_iteration > 1:
            port_output_name = output + '_IT_' + str(i + 1)

        output_port_img = transform_port_name_lvl(name=port_output_name, lvl=level)
        output_port_size = transform_port_size_lvl(lvl=level, rgb=False)
        input_port_list = [input_port]
        main_func_list = [input_port, wave_offset, kernel_b1.__str__(), kernel_b2.__str__(),
                          kernel_b3.__str__(), kernel_b4.__str__(), output_port_img]
        output_port_list = [(output_port_img, output_port_size, 'B', True)]

        job_name = job_name_create(action='Morph Thinning', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                                   Kernel_1=str(kernel_1_name), Kernel_2=str(kernel_2_name), Iteration=str(input_iteration))

        d = create_dictionary_element(job_module='morphological_operations',
                                      job_name=job_name,
                                      input_ports=input_port_list,
                                      max_wave=wave_offset,
                                      init_func_name='init_func', init_func_param=None,
                                      main_func_name='main_func_thinning',
                                      main_func_param=main_func_list,
                                      output_ports=output_port_list)

        jobs_dict.append(d)

        port_input_name = port_output_name

    return port_output_name


############################################################################################################################################
# Kernel processing jobs
############################################################################################################################################


def do_kernel_convolution_job(port_input_name: str, input_gx: str, input_gy: str,
                              port_output_name: str, job_name: str, is_rgb: bool = False,
                              level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> None:
    """
    The two kernels are convolved with the original image to calculate the approximations of the derivatives. Function for configure
    the kernel image calculation job. The implementation is done using OpenCV-image library.
    :param job_name: name you want for the job
    :param port_input_name: input image port name
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param input_gx: kernel for x to use. From kernels.py
    :param input_gy: kernel for y to use. From kernels.py
    :param port_output_name: name of output images port. One for Gx and Gx
    :param level:  pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: None
    """
    kernel_1 = input_gx
    kernel_2 = input_gy

    # check kernel passed
    if isinstance(input_gx, list):
        if kernel_1 not in custom_kernels_used:
            custom_kernels_used.append(kernel_1)
        kernel_1 = kernel_1.__str__()
    else:
        if not isinstance(input_gx, str):
            log_setup_info_to_console("CONVOLUTION JOB DIDN'T RECEIVE CORRECT KERNEL")
            return None

    if isinstance(input_gy, list):
        if kernel_2 not in custom_kernels_used:
            custom_kernels_used.append(kernel_2)
        kernel_2 = kernel_2.__str__()
    else:
        if not isinstance(input_gy, str):
            log_setup_info_to_console("CONVOLUTION JOB DIDN'T RECEIVE CORRECT KERNEL")
            return None

    port_name_gx_output = 'Gx_' + port_output_name
    port_name_gy_output = 'Gy_' + port_output_name

    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)
    output_port_name_gx = transform_port_name_lvl(name=port_name_gx_output, lvl=level)
    output_port_name_gy = transform_port_name_lvl(name=port_name_gy_output, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, output_port_name_gx, output_port_name_gy, kernel_1, kernel_2]
    output_port_list = [(output_port_name_gx, output_port_size, 'h', True), (output_port_name_gy, output_port_size, 'h', True)]

    job_name = job_name_create(action=job_name, input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='kernel_convolution',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_convolution',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_deriche_kernel_convolution_job(port_input_name: str, alpha: float, omega: float,
                                      port_output_name: str, job_name: str, is_rgb: bool = False,
                                      level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> None:
    """
    The Deriche filter is a smoothing filter (low-pass) which was designed to optimally detect, along with a derivation operator, the
    contours in an image (Canny criteria optimization). Besides, as this filter is very similar to a gaussian filter, but much simpler to
    implement (based on simple first order IIR filters), it is also much used for general image filtering. Indeed, contrary to a gaussian
    filter that is often implemented using a FIR (finite response) filter, and which complexity is directly dependant on the desired
    filtering level (standard deviation sigma), for a first order IIR, which equation is: y[n] = a*x[n] + (1-a)*y[n-1], the complexity
    is constant and very limited (2 multiplications per pixel), and the filtering level can be arbitrary modified through the "
    forget factor" a.
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.5736&rep=rep1&type=pdf
    :param job_name: name you want for the job
    :param port_input_name: input image port name
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param alpha:
    :param omega:
    :param port_output_name: name of output images port. One for Gx and Gx
    :param level:  pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: None
    """
    port_name_gx_output = 'Gx_' + port_output_name
    port_name_gy_output = 'Gy_' + port_output_name

    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)
    output_port_name_gx = transform_port_name_lvl(name=port_name_gx_output, lvl=level)
    output_port_name_gy = transform_port_name_lvl(name=port_name_gy_output, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, alpha, omega, output_port_name_gx, output_port_name_gy]
    output_port_list = [(output_port_name_gx, output_port_size, 'h', True), (output_port_name_gy, output_port_size, 'h', True)]

    job_name = job_name_create(action=job_name, input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='kernel_convolution',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_deriche_convolution',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_kernel_cross_convolution_job(job_name: str, port_input_name: str,
                                    kernel: str, port_output_name: str, wave_offset: int = 0,
                                    is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> None:
    """
    The eight kernels are convolved with the original image to calculate the approximations of the derivatives. Function for configure
    the kernel image calculation job. The job is added to the job buffer. The implementation is done using OpenCV-image library.
    :param job_name: name you want for the job
    :param port_input_name: input image port name
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: kernel to use
    :param port_output_name: name of output images port.
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: None
    """
    kernel_1 = kernel
    # check kernel passed
    if isinstance(kernel, list):
        if kernel_1 not in custom_kernels_used:
            custom_kernels_used.append(kernel_1)
        kernel_1 = kernel_1.__str__()
    else:
        if not isinstance(kernel, str):
            log_setup_info_to_console("CONVOLUTION JOB DIDN'T RECEIVE CORRECT KERNEL")
            return

    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)
    output_port_name = []

    for i in range(8):
        name = transform_port_name_lvl(name='G' + str(i) + '_' + port_output_name, lvl=level)
        output_port_name.append(name)

    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset]
    main_func_list.extend(output_port_name)
    main_func_list.append(kernel_1)

    output_port_list = []
    for el in output_port_name:
        output_port_list.append((el, output_port_size, 'i', True))

    job_name = job_name_create(action=job_name, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='kernel_convolution',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_cross',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_kernel_frei_chen_convolution_job(port_input_name: str,
                                        port_output_name: str = 'FREI_CHEN_3x3', dilated_kernel: int = 0,
                                        is_rgb: bool = False, wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> None:
    """
    The eight kernels are convolved with the original image to calculate the approximations of the derivatives. The implementation
    is done using OpenCV-image library.
    https://ieeexplore.ieee.org/document/1674733
    :param port_input_name: name you want for the job
    :param dilated_kernel: dilation factor of the kernel, by default is 0
    :param port_input_name: input image port name
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output images port.
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: None
    """

    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)
    output_port_name = []

    for i in range(9):
        name = transform_port_name_lvl(name='G' + str(i) + '_' + port_output_name + '_' + port_input_name, lvl=level)
        output_port_name.append(name)

    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, dilated_kernel]
    main_func_list.extend(output_port_name)

    output_port_list = []
    for el in output_port_name:
        output_port_list.append((el, output_port_size, 'i', True))

    job_name = job_name_create(action='Frei-Chen Convolution', input_list=[input_port_name], wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='kernel_convolution',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_frei_chen',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_kernel_navatia_babu_convolution_job(port_input_name: str,
                                           port_output_name: str = 'NAVATIA_BABU_5x5',
                                           wave_offset: int = 0, is_rgb: bool = False,
                                           level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> None:
    """
    The eight kernels are convolved with the original image to calculate the approximations of the derivatives. The implementation is
    done using OpenCV-image library.
    https://www.sciencedirect.com/science/article/abs/pii/0146664X80900490
    :param port_input_name: input image port name
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output images port.
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: None
    """

    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)
    output_port_name = []

    for i in range(6):
        name = transform_port_name_lvl(name='G' + str(i) + '_' + port_output_name + '_' + port_input_name, lvl=level)
        output_port_name.append(name)

    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)
    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset]
    main_func_list.extend(output_port_name)

    output_port_list = []
    for el in output_port_name:
        output_port_list.append((el, output_port_size, 'i', True))

    job_name = job_name_create(action='Convolution Navatia-Babu kernels', input_list=[port_input_name], wave_offset=[wave_offset],
                               level=level)

    d = create_dictionary_element(job_module='kernel_convolution',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_navatia_babu',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


############################################################################################################################################
# edge detection - magnitude gradient jobs
############################################################################################################################################


def do_gradient_magnitude_job(job_name: str, port_input_name_gx: str, port_input_name_gy: str,
                              port_output_name: str,
                              wave_offset_gx: int = 0, wave_offset_gy: int = 0, is_rgb: bool = False,
                              level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> None:
    """
    At each pixel in the image, the gradient approximations given by Gx and Gy are combined to give the gradient magnitude,
    using: sqrt(gx**2+gy**2). Function for configure the kernel image calculation job. The job is added to the job buffer.
    The implementation is done using OpenCV-image library.
    :param job_name: name you want for the job
    :param port_input_name_gx: kernel for Gx to use. From do_kernel_convolution_job
    :param wave_offset_gx: port wave offset. If 0 it is in current wave.
    :param port_input_name_gy: kernel for Gy to use. From do_kernel_convolution_job
    :param wave_offset_gy: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output image port.
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: None
    """
    input_port_name_gx = transform_port_name_lvl(name=port_input_name_gx, lvl=level)
    input_port_name_gy = transform_port_name_lvl(name=port_input_name_gy, lvl=level)

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name_gx, input_port_name_gy]
    main_func_list = [input_port_name_gx, wave_offset_gx, input_port_name_gy, wave_offset_gy, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action=job_name, wave_offset=[max(wave_offset_gx, wave_offset_gy)], level=level)

    d = create_dictionary_element(job_module='edge_gradient_magnitude',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=max(wave_offset_gx, wave_offset_gy),
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_gradient_orientation_job(job_name: str, port_input_name_gx: str, port_input_name_gy: str,
                                port_output_name: str,
                                wave_offset_gx: int = 0, wave_offset_gy: int = 0, is_rgb: bool = False,
                                level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0):
    """
    At each pixel in the image, the gradient approximations given by Gx and Gy are combined to give the gradient magnitude, using:
    sqrt(gx**2+gy**2). Function for configure the kernel image calculation job. The job is added to the job buffer. The implementation is
    done using OpenCV-image library.
    :param job_name: name you want for the job
    :param port_input_name_gx: kernel for Gx to use. From do_kernel_convolution_job
    :param wave_offset_gx: port wave offset. If 0 it is in current wave.
    :param port_input_name_gy: kernel for Gy to use. From do_kernel_convolution_job
    :param wave_offset_gy: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output image port.
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: None
    """
    input_port_name_gx = transform_port_name_lvl(name=port_input_name_gx, lvl=level)
    input_port_name_gy = transform_port_name_lvl(name=port_input_name_gy, lvl=level)

    output_port_orientation = transform_port_name_lvl(name=port_output_name + '_ANGLE', lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name_gx, input_port_name_gy]
    main_func_list = [input_port_name_gx, wave_offset_gx, input_port_name_gy, wave_offset_gy, output_port_orientation]
    output_port_list = [(output_port_orientation, output_port_size, 'f', True)]

    job_name = job_name_create(action=job_name, wave_offset=[max(wave_offset_gx, wave_offset_gy)], level=level)

    d = create_dictionary_element(job_module='edge_gradient_magnitude',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=max(wave_offset_gx, wave_offset_gy),
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_orientation',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_first_order_derivative_operators(port_input_name: str, operator: str, wave_offset: int = 0,
                                        level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, save_angle: bool = False,
                                        port_output_name: str = None, is_rgb: bool = False) -> str:
    """
    Function for configure the first derivative order edge detection operators. The implementation is done using OpenCV-image library.
    :param port_input_name: name you want for the job
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param operator: operator to use
    :param port_output_name: name of output image port.
    :param save_angle: if we want to save a image with the angles of magnitude.
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    operator_job = operator.replace('_', ' ')
    kernel_x = operator.lower() + '_x'
    kernel_y = operator.lower() + '_y'

    output_port = operator

    do_kernel_convolution_job(job_name='Convolution Kernel ' + operator_job,
                              port_input_name=port_input_name,
                              wave_offset=wave_offset,
                              input_gx=kernel_x,
                              input_gy=kernel_y,
                              port_output_name=output_port + '_' + port_input_name,
                              level=level, is_rgb=is_rgb)

    if port_output_name is None:
        port_output_name = output_port + '_' + port_input_name

    do_gradient_magnitude_job(job_name=operator_job + ' on ' + port_input_name,
                              port_input_name_gx='Gx_' + output_port + '_' + port_input_name,
                              wave_offset_gx=0,
                              port_input_name_gy='Gy_' + output_port + '_' + port_input_name,
                              wave_offset_gy=0,
                              port_output_name=port_output_name,
                              level=level, is_rgb=is_rgb)
    if save_angle:
        do_gradient_orientation_job(job_name=operator_job + ' direction on ' + port_input_name,
                                    port_input_name_gx='Gx_' + output_port + '_' + port_input_name,
                                    wave_offset_gx=0,
                                    port_input_name_gy='Gy_' + output_port + '_' + port_input_name,
                                    wave_offset_gy=0,
                                    port_output_name=port_output_name,
                                    level=level, is_rgb=False)

    return port_output_name


############################################################################################################################################
# edge detection - directional gradient jobs
############################################################################################################################################


def do_gradient_magnitude_cross_job(job_name: str, port_input_name: str, port_output_name: str,
                                    is_rgb: bool = False, wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> None:
    """
    At each pixel in the image, the gradient approximations given by Gx and Gy are combined to give the gradient magnitude,
    using: sqrt(gx**2+gy**2). Function for configure the kernel image calculation job. The job is added to the job buffer.
    The implementation is done using OpenCV-image library.
    :param job_name: name you want for the job
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_input_name: kernel for use. From do_kernel_convolution_job
    :param port_output_name: name of output image port.
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: None
    """
    input_port_list = []

    for i in range(8):
        name = transform_port_name_lvl(name='G' + str(i) + '_' + port_input_name, lvl=level)
        input_port_list.append(name)

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    main_func_list = []
    main_func_list.extend(input_port_list)
    main_func_list.append(wave_offset)
    main_func_list.append(output_port_name)
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    d = create_dictionary_element(job_module='edge_directional_magnitude',
                                  job_name=job_name + ' ' + port_input_name + ' ' + str(level) + ' W-' + str(wave_offset),
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_cross',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_gradient_frei_chen_job(port_input_name: str,
                              port_output_edge_name: str = 'FREI_CHEN_EDGE_3x3', port_output_line_name: str = 'FREI_CHEN_LINE_3x3',
                              is_rgb: bool = False, wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    At each pixel in the image, the gradient approximations given by Gx and Gy are combined to give the gradient magnitude, using:
    sqrt(gx**2+gy**2). The implementation is done using OpenCV-image library.
    https://ieeexplore.ieee.org/document/1674733
    :param port_input_name: name you want for the job
    :param port_input_name: kernel for use. From do_kernel_convolution_job
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_edge_name: name of output edge image port.
    :param port_output_line_name: name of output line image port.
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: Tuple containing name of edge port and line port of output ports
    """
    input_port_list = []

    for i in range(9):
        name = transform_port_name_lvl(name='G' + str(i) + '_' + port_input_name, lvl=level)
        input_port_list.append(name)

    output_port_edge_name = transform_port_name_lvl(name=port_output_edge_name, lvl=level)
    output_port_line_name = transform_port_name_lvl(name=port_output_line_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    main_func_list = []
    main_func_list.extend(input_port_list)
    main_func_list.append(wave_offset)
    main_func_list.append(output_port_edge_name)
    main_func_list.append(output_port_line_name)
    output_port_list = [(output_port_edge_name, output_port_size, 'B', True), (output_port_line_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Frei-Chen 3x3', input_list=[port_input_name.replace('FREI_CHEN_3x3_', '')],
                               wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='edge_directional_magnitude',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_frei_chen',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_edge_name, port_output_line_name


def do_gradient_navatia_babu_job(port_input_name: str,
                                 port_output_name: str = 'NAVATIA_BABU_5x5',
                                 is_rgb: bool = False, wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    At each pixel in the image, the gradient approximations given by Gx and Gy are combined to give the gradient magnitude, using:
    sqrt(gx**2+gy**2). The implementation is done using OpenCV-image library.
    https://www.sciencedirect.com/science/article/abs/pii/0146664X80900490
    :param port_input_name: kernel for use. From do_kernel_convolution_job
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output image port.
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_list = []

    for i in range(6):
        name = transform_port_name_lvl(name='G' + str(i) + '_' + 'NAVATIA_BABU_5x5_' + port_input_name, lvl=level)
        input_port_list.append(name)

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    main_func_list = []
    main_func_list.extend(input_port_list)
    main_func_list.append(wave_offset)
    main_func_list.append(output_port_name)
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Navatia-Babu edge', input_list=[port_input_name], wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='edge_directional_magnitude',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_6_cross',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return output_port_name


def do_frei_chen_edge_job(port_input_name: str, dilated_kernel: int = 0,
                          port_output_edge_name: str = 'FREI_CHEN_EDGE_3x3', port_output_line_name: str = 'FREI_CHEN_LINE_3x3',
                          level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> tuple:
    """
    This job creates 2 jobs: do_kernel_convolution_job and do_gradient_magnitude_job for Frei-Chen 3x3.
    The implementation is done using OpenCV-image library.
    Frei and Chung-Ching Chen, “Fast Boundary Detection: A Generalization and a New Algorithm,”
    https://ieeexplore.ieee.org/document/1674733
    The jobs are added to the job buffer.
    :param port_input_name: name of the job
    :param dilated_kernel: factor of dilation of kernel, by default = 0
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_edge_name: name of output edge image port.
    :param port_output_line_name: name of output line image port.
    :param is_rgb: if the output ports is rgb, 3 channels
    :param level: pyramid level to calculate at
    :return: output image port name of edge and line
    """
    output_name_kernel = 'FREI_CHEN_3x3'
    port_output_edge_name = 'FREI_CHEN_EDGE_3x3'
    port_output_line_name = 'FREI_CHEN_LINE_3x3'

    if dilated_kernel == 1:
        output_name_kernel = 'FREI_CHEN_DILATED_5x5'
        port_output_edge_name = 'FREI_CHEN_EDGE_DILATED_5x5'
        port_output_line_name = 'FREI_CHEN_LINE_DILATED_5x5'
    elif dilated_kernel == 2:
        output_name_kernel = 'FREI_CHEN_DILATED_7x7'
        port_output_edge_name = 'FREI_CHEN_EDGE_DILATED_7x7'
        port_output_line_name = 'FREI_CHEN_LINE_DILATED_7x7'

    do_kernel_frei_chen_convolution_job(port_input_name=port_input_name, dilated_kernel=dilated_kernel,
                                        wave_offset=wave_offset,
                                        port_output_name=output_name_kernel,
                                        level=level, is_rgb=is_rgb)

    edge_name, line_name = do_gradient_frei_chen_job(port_input_name=output_name_kernel + '_' + port_input_name,
                                                     wave_offset=0,
                                                     port_output_edge_name=port_output_edge_name + '_' + port_input_name,
                                                     port_output_line_name=port_output_line_name + '_' + port_input_name,
                                                     level=level, is_rgb=is_rgb)

    return edge_name, line_name


def do_navatia_babu_edge_5x5_job(port_input_name: str,
                                 level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0,
                                 port_output_name: str = 'NAVATIA_BABU_5x5', is_rgb: bool = False) -> str:
    """
    Navatia-Babu
    https://sci-hub.tw/https://www.sciencedirect.com/science/article/abs/pii/0146664X80900490
    This job creates 2 jobs: do_kernel_convolution_job and do_gradient_magnitude_job for Navatia-Babu 3x3.
    The jobs are added to the job buffer. The implementation is done using OpenCV-image library.
    :param port_input_name: name of the input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of the output port
    :param level: pyramid level to calculate at
    :param is_rgb
    :return: output image port name of edge
    """

    do_kernel_navatia_babu_convolution_job(port_input_name=port_input_name,
                                           is_rgb=is_rgb, wave_offset=wave_offset,
                                           level=level)

    do_gradient_navatia_babu_job(port_input_name=port_input_name,
                                 is_rgb=is_rgb, wave_offset=0,
                                 port_output_name=port_output_name + '_' + port_input_name, level=level)

    return port_output_name + '_' + port_input_name


def do_compass_edge_job(port_input_name: str,
                        operator: str,
                        port_output_name: str = None,
                        is_rgb: bool = False, wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    Do compass first derivative edge detection job.
    :param port_input_name: name of the input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of the output port
    :param level: pyramid level to calculate at
    :param operator: what operator to use
    :param is_rgb: if is colored or greyscale
    :return: output image port
    """
    kernel_x = operator.lower() + '_x'
    if operator == FILTERS.ROBINSON_CROSS_3x3:
        operator_job = 'ROBINSON_CROSS_3x3'
    elif operator == FILTERS.ROBINSON_CROSS_DILATED_5x5:
        operator_job = 'ROBINSON_CROSS_DILATED_5x5'
    elif operator == FILTERS.ROBINSON_CROSS_DILATED_7x7:
        operator_job = 'ROBINSON_CROSS_DILATED_7x7'
    elif operator == FILTERS.ROBINSON_MODIFIED_CROSS_3x3:
        operator_job = 'ROBINSON_MODIFIED_CROSS_3x3'
    elif operator == FILTERS.ROBINSON_MODIFIED_CROSS_5x5:
        operator_job = 'ROBINSON_MODIFIED_CROSS_5x5'
    elif operator == FILTERS.ROBINSON_MODIFIED_CROSS_7x7:
        operator_job = 'ROBINSON_MODIFIED_CROSS_7x7'
    else:
        operator_job = operator

    if 'COMPASS' not in operator or 'CROSS' not in operator:
        operator = operator + '_CROSS'

    output_port = operator_job + '_' + port_input_name

    do_kernel_cross_convolution_job(job_name='Convolution Kernels Cross ' + operator_job.replace('_', ' ') + ' ' + port_input_name,
                                    port_input_name=port_input_name,
                                    wave_offset=wave_offset,
                                    kernel=kernel_x,
                                    port_output_name=output_port,
                                    level=level, is_rgb=is_rgb)

    if port_output_name is None:
        port_output_name = output_port

    do_gradient_magnitude_cross_job(job_name=operator_job.replace('_', ' ') + ' ' + port_input_name,
                                    port_input_name=output_port,
                                    wave_offset=0,
                                    port_output_name=port_output_name,
                                    is_rgb=is_rgb,
                                    level=level)

    return port_output_name

def do_kirsch_3x3_cross_job(port_input_name: str,
                            level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0,
                            port_output_name: str = 'KIRSCH_CROSS_3x3', is_rgb: bool = False) -> str:
    """
    Standard Kirsch Filter
    https://www.sciencedirect.com/science/article/abs/pii/0010480971900346?via%3Dihub
    This job creates 2 jobs: do_kernel_cross_convolution_job and do_gradient_magnitude_cross_job for Kirsch 3x3
    The jobs are added to the job buffer.
    :param port_input_name: name of the input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of the output port
    :param level: pyramid level to calculate at
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    convolution_output = port_input_name.upper() + '_' + port_output_name

    do_kernel_cross_convolution_job(job_name='Kirsch Kernel Convolution 3x3',
                                    port_input_name=port_input_name, wave_offset=wave_offset,
                                    kernel='kirsch_3x3_x', is_rgb=is_rgb,
                                    port_output_name=convolution_output, level=level)

    do_gradient_magnitude_cross_job(job_name='Kirsch 3x3', wave_offset=0,
                                    port_input_name=convolution_output, is_rgb=is_rgb,
                                    port_output_name=port_output_name + '_' + port_input_name, level=level)

    return port_output_name


def do_robinson_3x3_cross_job(port_input_name: str,
                              port_output_name: str = 'ROBINSON_CROSS_3x3',
                              level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> str:
    """
    Standard Robinson Compass Filter
    https://www.sciencedirect.com/science/article/abs/pii/S0146664X77800245
    This job creates 2 jobs: do_kernel_cross_convolution_job and do_gradient_magnitude_cross_job for Robinson 3x3
    The jobs are added to the job buffer.
    :param port_input_name: name of the input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of the output port
    :param level: pyramid level to calculate at
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    convolution_output = port_input_name.upper() + '_' + port_output_name

    do_kernel_cross_convolution_job(job_name='Robinson Kernel Convolution 3x3',
                                    port_input_name=port_input_name, wave_offset=wave_offset,
                                    kernel='sobel_3x3_x', is_rgb=is_rgb,
                                    port_output_name=convolution_output, level=level)

    do_gradient_magnitude_cross_job(job_name='Robinson Compass 3x3', wave_offset=0,
                                    port_input_name=convolution_output, is_rgb=is_rgb,
                                    port_output_name=port_output_name + '_' + port_input_name, level=level)

    return port_output_name


def do_robinson_modified_3x3_cross_job(port_input_name: str,
                                       port_output_name: str = 'ROBINSON_MODIFIED_CROSS_3x3',
                                       wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False) -> str:
    """
    Modified Robinson Compass Filter
    https://www.sciencedirect.com/science/article/abs/pii/S0146664X77800245
    This job creates 2 jobs: do_kernel_cross_convolution_job and do_gradient_magnitude_cross_job for Robinson 3x3
    The jobs are added to the job buffer.
    :param port_input_name: name of the input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of the output port
    :param level: pyramid level to calculate at
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    convolution_output = port_input_name.upper() + '_' + port_output_name

    do_kernel_cross_convolution_job(job_name='Robinson Modified Kernel Convolution 3x3',
                                    port_input_name=port_input_name, wave_offset=wave_offset,
                                    kernel='prewitt_3x3_x', is_rgb=is_rgb,
                                    port_output_name=convolution_output, level=level)

    do_gradient_magnitude_cross_job(job_name='Robinson Modified Compass 3x3', wave_offset=0,
                                    port_input_name=convolution_output, is_rgb=is_rgb,
                                    port_output_name=port_output_name + '_' + port_input_name, level=level)

    return port_output_name


def do_prewitt_3x3_cross_job(port_input_name: str, port_output_name: str = 'PREWITT_CROSS_3x3',
                             level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> str:
    """
    Standard Prewitt Compass Filter
    https://books.google.ro/books?hl=en&lr=&id=vp-w_pC9JBAC&oi=fnd&pg=PA75&dq=%22Object+Enhancement+and+Extraction%22&ots=szJ80qlBD5&sig=5qT-zX7eoMUnEa4YiQ4wN9rPeDg&redir_esc=y#v=onepage&q=%22Object%20Enhancement%20and%20Extraction%22&f=false
    This job creates 2 jobs: do_kernel_cross_convolution_job and do_gradient_magnitude_cross_job for Kirsch 3x3
    The jobs are added to the job buffer.
    :param port_input_name: name of the input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of the output port
    :param level: pyramid level to calculate at
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    convolution_output = port_input_name.upper() + '_' + port_output_name

    do_kernel_cross_convolution_job(job_name='Prewitt Kernel Convolution 3x3', wave_offset=wave_offset,
                                    port_input_name=port_input_name,
                                    kernel='prewitt_compass_3x3_x', is_rgb=is_rgb,
                                    port_output_name=convolution_output, level=level)

    do_gradient_magnitude_cross_job(job_name='Prewitt Compass 3x3', wave_offset=0,
                                    port_input_name=convolution_output, is_rgb=is_rgb,
                                    port_output_name=port_output_name + '_' + port_input_name, level=level)

    return port_output_name


############################################################################################################################################
# edge detection - Canny jobs
############################################################################################################################################


def do_canny_from_kernel_convolution_job(kernel_convolution: str, port_output_name: str,
                                         config_canny_threshold: str, config_canny_threshold_value: str,
                                         wave_offset: int = 0, low_manual_threshold=None, high_manual_threshold=None,
                                         level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0):
    """
    Apply non-maximum suppression to get rid of spurious response to edge detection. Apply double threshold to determine potential edges.
    Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong
    edges. The implementation is done using OpenCV-image library.
    # https://www.cs.bgu.ac.il/~icbv161/wiki.files/Readings/1986-Canny-A_Computational_Approach_to_Edge_Detection.pdf
    :param kernel_convolution: name of gx and gy convolution results
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param config_canny_threshold: configuration of canny for thresholds
    :param config_canny_threshold_value: Value parameter for certain configurations
            None if not needed
    :param port_output_name: name of port you desire
    :param low_manual_threshold: manual set low threshold
    :param high_manual_threshold: manual set high threshold
    :param level: pyramid level to calculate at
    :return: None
    """
    kernel_convolution = kernel_convolution.replace('X', 'x')
    input_port_name_gx = transform_port_name_lvl(name='Gx_' + kernel_convolution, lvl=level)
    input_port_name_gy = transform_port_name_lvl(name='Gy_' + kernel_convolution, lvl=level)

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    value_port_name = config_canny_threshold_value

    if config_canny_threshold_value is not None:
        value_port_name = transform_port_name_lvl(name=config_canny_threshold_value, lvl=level)

    input_port_list = [input_port_name_gx, input_port_name_gy]
    main_func_list = [input_port_name_gx, input_port_name_gy, wave_offset, output_port_name, config_canny_threshold,
                      low_manual_threshold, high_manual_threshold, value_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Canny', input_list=[input_port_name_gx.replace('Gx_', '')], wave_offset=[wave_offset], level=level,
                               CONFIG=config_canny_threshold, VALUE=value_port_name)

    d = create_dictionary_element(job_module='edge_canny_cv2',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_var_trh',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_canny_config_job(port_input_name: str,
                        edge_detector: str, canny_config: str, canny_config_value: str,
                        port_output_name: str = None, low_manual_threshold=None, high_manual_threshold=None,
                        do_blur: bool = True, kernel_blur_size: int = 3, sigma: float = 1,
                        do_otsu: bool = False,
                        level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    This job will add actually 3 jobs to the buffer:
    do_gaussian_blur_image_job, do_kernel_convolution_job, do_canny_from_kernel_convolution_job
    The Process of Canny edge detection algorithm can be broken down to 5 different steps:
    # https://www.cs.bgu.ac.il/~icbv161/wiki.files/Readings/1986-Canny-A_Computational_Approach_to_Edge_Detection.pdf
    Apply Gaussian filter to smooth the image in order to remove the noise
    Find the intensity gradients of the image
    Apply non-maximum suppression to get rid of spurious response to edge detection
    Apply double threshold to determine potential edges
    Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
    :param port_input_name: name of port on which we desire to apply Canny on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param edge_detector: which edge operator we want to use
    :param low_manual_threshold: manual set low threshold
    :param high_manual_threshold: manual set high threshold
    :param canny_config: configuration of canny for threshold
        fix threshold of low = 60, high = 90
        FIX_THRESHOLD = 0
        threshold high = otsu and low = otsu * 0.5
        OTSU_HALF = 1
        threshold calculation using sigma and median
        OTSU_MEDIAN_SIGMA = 2
        threshold calculation using sigma and median
        MEDIAN_SIGMA = 3
        threshold calculation ratio
        RATIO_THRESHOLD = 4
        threshold calculation ratio
        RATIO_MEAN = 5
    :param canny_config_value: Value parameter for certain configurations
            None if not needed
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param do_blur: if we want to do blur before canny
    :param kernel_blur_size: size of laplace kernel for blur
    :param sigma: sigma value for blur
    :param do_otsu: if you want to use otsu value for canny
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    kernels_ports = edge_detector.lower().replace(' ', '_')
    gx_kernel = kernels_ports + '_x'
    gy_kernel = kernels_ports + '_y'
    convolution_output = kernels_ports.upper().replace('X', 'x')
    wave = wave_offset

    if do_blur:
        # Gaussian Blur jobs
        output_name_blur = 'GAUS_BLUR_K_' + str(kernel_blur_size) + '_S_' + str(sigma).replace('.', '_') + '_' + port_input_name
        do_gaussian_blur_image_job(port_input_name=port_input_name,
                                   port_output_name=output_name_blur, level=level,
                                   kernel_size=kernel_blur_size,
                                   sigma=sigma, is_rgb=is_rgb, wave_offset=wave)
        wave = 0
        port_input_name = output_name_blur

    if do_otsu:
        input_otsu = port_input_name
        wave_otsu = wave_offset
        if is_rgb is True:
            output = 'GRAY_' + input_otsu
            do_grayscale_transform_job(port_input_name=port_input_name, wave_offset=wave_otsu,
                                       port_output_name=output)
            input_otsu = output
            wave_otsu = 0

        do_otsu_job(port_input_name=input_otsu,
                    port_output_name='OTSU_' + input_otsu,
                    wave_offset=wave_otsu, level=level)
        canny_config_value = 'OTSU_' + input_otsu + '_VALUE'

    convolution_output = convolution_output + '_' + port_input_name.upper()

    operator_job = edge_detector.replace('_', ' ')

    # Do gradient gx and gy for Sobel like an example
    do_kernel_convolution_job(job_name='Convolution Kernel ' + operator_job,
                              port_input_name=port_input_name,
                              input_gx=gx_kernel,
                              input_gy=gy_kernel,
                              port_output_name=convolution_output, wave_offset=wave,
                              level=level, is_rgb=is_rgb)
    wave = 0

    if port_output_name is None:
        if canny_config_value is not None:
            port_output_name = 'CANNY_' + canny_config.split('.')[-1] + '_' + edge_detector + '_' + port_input_name
        elif low_manual_threshold is not None and high_manual_threshold is not None:
            port_output_name = 'CANNY_' + canny_config.split('.')[-1] + '_' + edge_detector + '_' + str(low_manual_threshold) + '_' \
                               + str(high_manual_threshold) + '_' + port_input_name
            canny_config = CANNY_VARIANTS.MANUAL_THRESHOLD
        else:
            port_output_name = 'CANNY_' + canny_config.split('.')[-1] + '_' + edge_detector + '_80_170_' + port_input_name

    # Do Canny from precalculated intensity gradients
    do_canny_from_kernel_convolution_job(kernel_convolution=convolution_output,
                                         config_canny_threshold=canny_config, wave_offset=wave,
                                         config_canny_threshold_value=canny_config_value,
                                         low_manual_threshold=low_manual_threshold,
                                         high_manual_threshold=high_manual_threshold,
                                         port_output_name=port_output_name, level=level)

    return port_output_name


# noinspection PyTypeChecker
def do_canny_fix_threshold_job(port_input_name: str,
                               edge_detector: str = FILTERS.SOBEL_3x3, port_output_name: str = None,
                               low_manual_threshold=None, high_manual_threshold=None,
                               do_blur: bool = True, kernel_blur_size: int = 3, sigma: float = 1,
                               level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> str:
    """
    This job will add actually 3 jobs to the buffer:
    do_gaussian_blur_image_job, do_kernel_convolution_job, do_canny_from_kernel_convolution_job
    This job will set canny_config='CANNY_CONFIG.FIX_THRESHOLD', canny_config_value=None, by default
    The Process of Canny edge detection algorithm can be broken down to 5 different steps:
    # https://www.cs.bgu.ac.il/~icbv161/wiki.files/Readings/1986-Canny-A_Computational_Approach_to_Edge_Detection.pdf
    Apply Gaussian filter to smooth the image in order to remove the noise
    Find the intensity gradients of the image
    Apply non-maximum suppression to get rid of spurious response to edge detection
    Apply double threshold to determine potential edges
    Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
    :param port_input_name: name of port on which we desire to apply Canny on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param edge_detector: which edge operator we want to use
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param low_manual_threshold: manual set low threshold
    :param high_manual_threshold: manual set high threshold
    :param do_blur: if we want to do blur before canny
    :param kernel_blur_size size of laplace kernel for blur
    :param sigma sigma value for blur
    :param is_rgb: if is colored or greyscale
      :return: output image port name of edge and line
    """
    return do_canny_config_job(port_input_name=port_input_name, edge_detector=edge_detector,
                               canny_config=CANNY_VARIANTS.FIX_THRESHOLD, canny_config_value=None,
                               port_output_name=port_output_name, level=level, wave_offset=wave_offset, is_rgb=is_rgb,
                               low_manual_threshold=low_manual_threshold, high_manual_threshold=high_manual_threshold,
                               do_blur=do_blur, kernel_blur_size=kernel_blur_size, sigma=sigma)


def do_canny_ratio_threshold_job(port_input_name: str, canny_config_value: str, edge_detector: str = FILTERS.SOBEL_3x3,
                                 is_rgb: bool = False, port_output_name: str = None, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0,
                                 do_blur: bool = True, kernel_blur_size: int = 0, sigma: float = 1, wave_offset: int = 0) -> str:
    """
    This job will add actually 3 jobs to the buffer:
    do_gaussian_blur_image_job, do_kernel_convolution_job, do_canny_from_kernel_convolution_job
    This job will set canny_config='CANNY_CONFIG.RATIO_THRESHOLD' by default
    The Process of Canny edge detection algorithm can be broken down to 5 different steps:
    # http://justin-liang.com/tutorials/canny/
    # https://ieeexplore.ieee.org/document/5739265
    Apply Gaussian filter to smooth the image in order to remove the noise
    Find the intensity gradients of the image
    Apply non-maximum suppression to get rid of spurious response to edge detection
    Apply double threshold to determine potential edges
    Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
    :param port_input_name: name of port on which we desire to apply Canny on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param edge_detector: which edge operator we want to use
    :param canny_config_value: port on which the value should be taken
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param do_blur: if we want to do blur before canny
    :param kernel_blur_size size of laplace kernel for blur
    :param sigma sigma value for blur
    :param is_rgb: if is colored or greyscale
    :return: output image port name of edge and line
    """
    return do_canny_config_job(port_input_name=port_input_name, edge_detector=edge_detector,
                               canny_config=CANNY_VARIANTS.RATIO_THRESHOLD, canny_config_value=canny_config_value,
                               port_output_name=port_output_name, level=level, wave_offset=wave_offset, is_rgb=is_rgb,
                               do_blur=do_blur, kernel_blur_size=kernel_blur_size, sigma=sigma)


def do_canny_median_sigma_job(port_input_name: str, canny_config_value: str,
                              edge_detector: str = FILTERS.SOBEL_3x3, port_output_name: str = None,
                              do_blur: bool = True, kernel_blur_size: int = 3, sigma: float = 1,
                              level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    This job will add actually 3 jobs to the buffer:
    do_gaussian_blur_image_job, do_kernel_convolution_job, do_canny_from_kernel_convolution_job
    This job will set canny_config='CANNY_CONFIG.OTSU_MEDIAN_SIGMA' by default
    The Process of Canny edge detection algorithm can be broken down to 5 different steps:
    # https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    Apply Gaussian filter to smooth the image in order to remove the noise
    Find the intensity gradients of the image
    Apply non-maximum suppression to get rid of spurious response to edge detection
    Apply double threshold to determine potential edges
    Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
    :param port_input_name: name of port on which we desire to apply Canny on
    :param edge_detector: which edge operator we want to use
    :param canny_config_value: port on which the value should be taken
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param do_blur: if we want to do blur before canny
    :param kernel_blur_size size of laplace kernel for blur
    :param sigma sigma value for blur
    :param is_rgb: if is colored or greyscale
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name of edge and line
    """
    return do_canny_config_job(port_input_name=port_input_name, edge_detector=edge_detector,
                               canny_config=CANNY_VARIANTS.MEDIAN_SIGMA, canny_config_value=canny_config_value,
                               port_output_name=port_output_name, level=level, is_rgb=is_rgb, wave_offset=wave_offset,
                               do_blur=do_blur, kernel_blur_size=kernel_blur_size, sigma=sigma)


def do_canny_mean_sigma_job(port_input_name: str, canny_config_value: str,
                            edge_detector: str = FILTERS.SOBEL_3x3, port_output_name: str = None,
                            do_blur: bool = True, kernel_blur_size: int = 3, sigma: float = 1,
                            level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    This job will add actually 3 jobs to the buffer:
    do_gaussian_blur_image_job, do_kernel_convolution_job, do_canny_from_kernel_convolution_job
    This job will set canny_config='CANNY_CONFIG.OTSU_MEDIAN_SIGMA' by default
    The Process of Canny edge detection algorithm can be broken down to 5 different steps:
    # http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
    Apply Gaussian filter to smooth the image in order to remove the noise
    Find the intensity gradients of the image
    Apply non-maximum suppression to get rid of spurious response to edge detection
    Apply double threshold to determine potential edges
    Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
    :param port_input_name: name of port on which we desire to apply Canny on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param edge_detector: which edge operator we want to use
    :param canny_config_value: port on which the value should be taken
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param do_blur: if we want to do blur before canny
    :param kernel_blur_size size of laplace kernel for blur
    :param sigma sigma value for blur
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    return do_canny_config_job(port_input_name=port_input_name, edge_detector=edge_detector,
                               canny_config=CANNY_VARIANTS.RATIO_MEAN, canny_config_value=canny_config_value,
                               port_output_name=port_output_name, level=level, is_rgb=is_rgb, wave_offset=wave_offset,
                               do_blur=do_blur, kernel_blur_size=kernel_blur_size, sigma=sigma)


def do_canny_otsu_half_job(port_input_name: str,
                           edge_detector: str = FILTERS.SOBEL_3x3, port_output_name: str = None,
                           level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False,
                           do_blur: bool = True, kernel_blur_size: int = 3, sigma: float = 1) -> str:
    """
    This job will add actually 3 jobs to the buffer:
    do_gaussian_blur_image_job, do_kernel_convolution_job, do_canny_from_kernel_convolution_job
    This job will set canny_config='CANNY_CONFIG.OTSU_HALF' by default
    The Process of Canny edge detection algorithm can be broken down to 5 different steps:
    Apply Gaussian filter to smooth the image in order to remove the noise
    Find the intensity gradients of the image
    Apply non-maximum suppression to get rid of spurious response to edge detection
    Apply double threshold to determine potential edges
    Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.403.5666&rep=rep1&type=pdf#page=120
    :param port_input_name: name of port on which we desire to apply Canny on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param edge_detector: which edge operator we want to use
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param do_blur: if we want to do blur before canny
    :param kernel_blur_size: size of laplace kernel for blur
    :param sigma: sigma value for blur
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    return do_canny_config_job(port_input_name=port_input_name, edge_detector=edge_detector,
                               canny_config=CANNY_VARIANTS.OTSU_HALF, canny_config_value='',
                               port_output_name=port_output_name, level=level, wave_offset=wave_offset,
                               do_blur=do_blur, kernel_blur_size=kernel_blur_size, sigma=sigma,
                               do_otsu=True, is_rgb=is_rgb)


def do_canny_otsu_median_sigma_job(port_input_name: str,
                                   edge_detector: str = FILTERS.SOBEL_3x3, port_output_name: str = None,
                                   do_blur: bool = True, kernel_blur_size: int = 3, sigma: float = 1,
                                   level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False):
    """
    This job will add actually 3 jobs to the buffer:
    do_gaussian_blur_image_job, do_kernel_convolution_job, do_canny_from_kernel_convolution_job
    This job will set canny_config='CANNY_CONFIG.OTSU_MEDIAN_SIGMA' by default
    The Process of Canny edge detection algorithm can be broken down to 5 different steps:
    # https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    Apply Gaussian filter to smooth the image in order to remove the noise
    Find the intensity gradients of the image
    Apply non-maximum suppression to get rid of spurious response to edge detection
    Apply double threshold to determine potential edges
    Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
    :param port_input_name: name of port on which we desire to apply Canny on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param edge_detector: which edge operator we want to use
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param do_blur: if we want to do blur before canny
    :param kernel_blur_size: size of laplace kernel for blur
    :param sigma: sigma value for blur
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    return do_canny_config_job(port_input_name=port_input_name, edge_detector=edge_detector,
                               canny_config=CANNY_VARIANTS.OTSU_MEDIAN_SIGMA, canny_config_value='',
                               port_output_name=port_output_name, level=level, wave_offset=wave_offset,
                               do_blur=do_blur, kernel_blur_size=kernel_blur_size, sigma=sigma,
                               do_otsu=True, is_rgb=is_rgb)


def do_deriche_canny_job(port_input_name: str, alpha: float, omega: float, canny_config: str, canny_config_value: str = None,
                         port_output_name: str = None, low_manual_threshold=None, high_manual_threshold=None,
                         level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    Deriche edge detector is an edge detection operator developed by Rachid Deriche in 1987. It's a multistep algorithm used to obtain an
    optimal result of edge detection in a discrete two-dimensional image. This algorithm is based on John F. Canny's work related to the
    edge detection (Canny's edge detector) and his criteria for optimal edge detection.
    Find the intensity gradients of the image
    Apply non-maximum suppression to get rid of spurious response to edge detection
    Apply double threshold to determine potential edges
    Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.5736&rep=rep1&type=pdf
    :param port_input_name: name of port on which we desire to apply Canny on
    :param alpha:
    :param omega:
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param low_manual_threshold: manual set low threshold
    :param high_manual_threshold: manual set high threshold
    :param canny_config: configuration of canny for threshold
        fix threshold of low = 60, high = 90
        FIX_THRESHOLD = 0
        threshold high = otsu and low = otsu * 0.5
        OTSU_HALF = 1
        threshold calculation using sigma and median
        OTSU_MEDIAN_SIGMA = 2
        threshold calculation using sigma and median
        MEDIAN_SIGMA = 3
        threshold calculation ratio
        RATIO_THRESHOLD = 4
        threshold calculation ratio
        RATIO_MEAN = 5
    :param canny_config_value: Value parameter for certain configurations
            None if not needed
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    convolution_output = 'DERICHE_CONVOLUTION'
    # Do gradient gx and gy for Sobel like an example
    do_deriche_kernel_convolution_job(job_name='Convolution Deriche Kernel ',
                                      port_input_name=port_input_name,
                                      alpha=alpha, omega=omega,
                                      port_output_name=convolution_output, wave_offset=wave_offset,
                                      level=level, is_rgb=is_rgb)
    wave = 0

    if port_output_name is None:
        if canny_config_value is not None:
            port_output_name = 'CANNY_' + canny_config.split('.')[-1] + '_DERICHE_' + port_input_name
        elif low_manual_threshold is not None and high_manual_threshold is not None:
            port_output_name = 'CANNY_' + canny_config.split('.')[-1] + '_DERICHE_' + str(low_manual_threshold) + '_' \
                               + str(high_manual_threshold) + '_' + port_input_name
            canny_config = CANNY_VARIANTS.MANUAL_THRESHOLD
        else:
            port_output_name = 'CANNY_' + canny_config.split('.')[-1] + '_DERICHE_80_170_' + port_input_name

    # Do Canny from precalculated intensity gradients
    do_canny_from_kernel_convolution_job(kernel_convolution=convolution_output,
                                         config_canny_threshold=canny_config, wave_offset=wave,
                                         config_canny_threshold_value=canny_config_value,
                                         low_manual_threshold=low_manual_threshold,
                                         high_manual_threshold=high_manual_threshold,
                                         port_output_name=port_output_name, level=level)

    return port_output_name


############################################################################################################################################
# edge detection - second derivative
############################################################################################################################################
# noinspection PyTypeChecker
def do_laplace_job(port_input_name: str, threshold_value: int = 0,
                   kernel: str = FILTERS_SECOND_ORDER.LAPLACE_1, port_output_name: str = None,
                   level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> str:
    """
    Laplacian Operator is also a derivative operator which is used to find edges in an image.The second derivative of a
    smoothed step edge is a function that crosses. The Laplacian is the two-dimensional equivalent of the second derivative.
    # https://www.sciencedirect.com/science/article/abs/pii/0734189X8990131X
    :param port_input_name: name of port on which we desire to apply Laplace operator on
    :param threshold_value: value of threshold the edge pixel values [0-255]
                            If 0 no threshold is applied
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel: which kernel we want to use
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    # check kernel passed
    if isinstance(kernel, list):
        if kernel not in custom_kernels_used:
            custom_kernels_used.append(kernel)
        kernel = kernel.__str__()
    else:
        if not isinstance(kernel, str):
            log_setup_info_to_console("CONVOLUTION JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            kernel = kernel.lower() + '_xy'

    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = str((kernel.upper().replace('_XY', '')).replace('X', 'x')) + '_' + port_input_name

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port]
    main_func_list = [input_port, wave_offset, threshold_value, kernel, output_port]
    output_port_list = [(output_port, output_port_size, 'h', True)]

    job_name = job_name_create(action='Laplace Edge Operator', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='edge_second_order',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_laplace',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_laplacian_pyramid_from_img_diff_job(port_input_name_1: str, port_input_name_2: str,
                                           port_output_name: str = None,
                                           level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> str:
    """
    Do laplace edge detection using difference from 2 matrices or images, this functions uses opencv. The algorithm can also be used to
    obtain an approximation of the Laplacian of Gaussian when the ratio of s(2) to s(1) is roughly equal to 1.6. The Laplacian of
    Gaussian is useful for detecting edges that appear at various image scales or degrees of image focus. The exact values of s(1) and
    s(2) that are used to approximate the Laplacian of Gaussian will determine the scale of the difference image, which may appear
    blurry as a result.
    # https://royalsocietypublishing.org/doi/10.1098/rspb.1980.0020
    :param port_input_name_1: first matrix
    :param port_input_name_2: second matrix
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param level:  pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name_1, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_input_name_2, lvl=level)

    if port_output_name is None:
        port_output_name = 'LAPLACE_PYRAMID_' + port_input_name_1

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_1, input_port_2]
    main_func_list = [input_port_1, input_port_2, wave_offset, output_port]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Laplace Pyramid', input_list=input_port_list, wave_offset=[wave_offset, wave_offset], level=level)

    d = create_dictionary_element(job_module='edge_second_order',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_laplacian_pyramid_2_images',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return output_port


def do_laplacian_from_img_diff_job(port_original_input_name: str, port_smoothed_input_name: str,
                                   do_binary: bool = False, port_output_name: str = None,
                                   level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> str:
    """
    Do laplace edge detection using difference from 2 matrices or images, this functions uses opencv.
    :param port_original_input_name: first matrix
    :param port_smoothed_input_name: second matrix
    :param do_binary: does the image into 0 or 1
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param level:  pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_original_input_name, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_smoothed_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'BAND_LIMITED_LAPLACE_' + port_original_input_name

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_1, input_port_2]
    main_func_list = [input_port_1, input_port_2, wave_offset, output_port, do_binary]
    output_port_list = [(output_port, output_port_size, 'h', True)]

    job_name = job_name_create(action='Band limited Laplace', input_list=input_port_list, wave_offset=[wave_offset, wave_offset],
                               level=level)

    d = create_dictionary_element(job_module='edge_second_order',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_binary_laplace_2_images',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return output_port


def do_log_job(port_input_name: str,
               gaussian_kernel_size: int = 0, gaussian_sigma = 0.0,
               laplacian_kernel: str = FILTERS_SECOND_ORDER.LAPLACE_1, use_precalculated_kernel: bool = False,
               port_output_name: str = None,
               level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    Since derivative filters are very sensitive to noise, it is common to smooth the image (e.g., using a Gaussian filter) before applying
    the Laplacian. This two-step process is call the Laplacian of Gaussian (LoG) operation. The LoG operator takes the second derivative
    of the image. Where the image is basically uniform, the LoG will give zero. Wherever a change occurs, the LoG will give a positive
    response on the darker side and a negative response on the lighter side.  The implementation is done using OpenCV-image library.
    # https://ieeexplore.ieee.org/abstract/document/4767838
    :param port_input_name: name of port on which we desire to apply Canny on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param gaussian_kernel_size: which kernel we want to use for gaussian blur
    :param gaussian_sigma: which sigma we want to use for gaussian blur
    :param laplacian_kernel: which kernel we want to use for laplace operator
    :param use_precalculated_kernel: if the laplace kernel is calculated with the gaussian blur
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    wave = wave_offset
    port_output_name_2 = 'LOG_' + port_input_name

    if use_precalculated_kernel is False:
        output_name_blur = 'GAUSS_BLUR_K_' + str(gaussian_kernel_size) + '_S_' + str(gaussian_sigma).replace('.', '_') \
                           + '_' + port_input_name
        port_output_name_2 = 'LOG_' + str((laplacian_kernel.upper().replace('_XY', '')).replace('X', 'x')) + '_'+ output_name_blur
        do_gaussian_blur_image_job(port_input_name=port_input_name, wave_offset=wave,
                                   port_output_name=output_name_blur, level=level, is_rgb=is_rgb,
                                   kernel_size=gaussian_kernel_size, sigma=gaussian_sigma)
        wave = 0
        port_input_name = output_name_blur

    if port_output_name is None:
        port_output_name = port_output_name_2

    temp = do_laplace_job(port_input_name=port_input_name, kernel=laplacian_kernel,
                          port_output_name=port_output_name, is_rgb=is_rgb, wave_offset=wave)

    return temp


def do_zero_crossing_job(port_input_name: str,
                         threshold: float = 0.3, port_output_name: str = None,
                         level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0, is_rgb: bool = False) -> str:
    """
    First convert the LOG-convolved image to a binary image, by replacing the pixel values by 1 for positive values and 0 for negative
    values. In order to compute the zero crossing pixels, we need to simply look at the boundaries of the non-zero regions in this binary
    image. Boundaries can be found by finding any non-zero pixel that has an immediate neighbor which is is zero.
    Hence, for each pixel, if it is non-zero, consider its 8 neighbors, if any of the neighboring pixels is zero,
    the pixel can be identified as an edge.
    # https://royalsocietypublishing.org/doi/abs/10.1098/rspb.1980.0020
    Works only on grayscale.
    :param port_input_name: input image
    :param threshold: thresholding value between 0 - 1
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param level:  pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'ZERO_CROSSING_THR_' + str(int(threshold * 255)) + '_' + port_input_name

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_1]
    main_func_list = [input_port_1, wave_offset, threshold, output_port]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Zero Crossing', input_list=input_port_list, wave_offset=[wave_offset], level=level,
                               THR=int(threshold * 255))

    d = create_dictionary_element(job_module='edge_second_order',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_zero_crossing',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_zero_crossing_adaptive_window_isef_job(port_original_input_name: str, port_bli_input_name: str, port_smoothed_input_name: str,
                                              win_size: int = 7, port_output_name: str = None,
                                              level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False,
                                              wave_offset: int = 0) -> str:
    """
    First convert the LOG-convolved image to a binary image, by replacing the pixel values by 1 for positive values and 0 for negative values.
    In order to compute the zero crossing pixels, we need to simply look at the boundaries of the non-zero regions in this binary image.
    Works only on grayscale.
    :param port_original_input_name: original input image
    :param port_bli_input_name: Binary Laplace image of original one
    :param port_smoothed_input_name: Smoothed image of original one
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: result matrix
    :param win_size: window size of adaptive filter to use
    :param level:  pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_original_input_name, lvl=level)
    input_port_2 = transform_port_name_lvl(name=port_smoothed_input_name, lvl=level)
    input_port_3 = transform_port_name_lvl(name=port_bli_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'ZERO_CROSSING_ADAPTIVE_WIN_' + port_original_input_name

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_1, input_port_2, input_port_3]
    main_func_list = [input_port_1, input_port_2, input_port_3, wave_offset, win_size, output_port]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Zero Crossing Adaptive', input_list=input_port_list,
                               wave_offset=[wave_offset, wave_offset, wave_offset],
                               level=level, WIN_SIZE=win_size)

    d = create_dictionary_element(job_module='edge_shen_castan',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_zero_crossing_adaptive',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_threshold_hysteresis_isef_job(port_input_name: str,
                                     ratio: float = 0.8, thinning_factor: float = 1,
                                     port_output_name: str = None,
                                     level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> str:
    """
    First convert the LOG-convolved image to a binary image, by replacing the pixel values by 1 for positive values and 0 for negative values.
    In order to compute the zero crossing pixels, we need to simply look at the boundaries of the non-zero regions in this binary image.
    Works only on grayscale.
    :param port_input_name: original input image
    :param ratio: This parameter specifies the percents of pixels above the hysteresis thresholding.
    :param thinning_factor: Thinning factor (number of pixels). See the description and examples above for further information and a
    possible value.
    :param port_output_name: result matrix
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param level:  pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    input_port_1 = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'ISEF_' + port_input_name

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_1]
    main_func_list = [input_port_1, wave_offset, ratio, thinning_factor, output_port]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='ISEF', input_list=input_port_list, wave_offset=[wave_offset, wave_offset], level=level)

    d = create_dictionary_element(job_module='edge_shen_castan',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_thr_hysteresis',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_shen_castan_job(port_input_name: str, laplacian_kernel: str = None, laplacian_threhold: int = 0,
                       smoothing_factor: float = 0.9, ratio: float = 0.9, thinning_factor: float = 0.5, zc_window_size: int = 7,
                       port_output_name: str = None,
                       level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> str:
    """
    This function will find edges on the input image by the Shen-Castan edge detection method. This function works for 8 and 24 bit per
    pixel images. Shen-Castan edge detector uses an optima filter function called Infinite Symmetric Exponential Filter (ISEF).
    This produces better signal to noise ratios and better localisation than Canny. First step is convolving the input image with the ISEF,
    then localization edges by zero crossing of the Laplacian (similar to the Marr-Hildreth algorithm). The approximation of Laplacian is
    computed by subtracting the original image from the smoothed one. The result is a band-limited Laplacian image. Next, a binary
    Laplacian image is generated by setting all the positive valued pixels to 1 and all others to 0. The candidate pixels are on the
    boundaries of the regions in th binary image. After improving the quality of edge pixels by false zero-crossing suppression,
    adaptive gradient thresholding and a hysteresis thresholding is applied finally.
    :param port_input_name: original input image
    :param smoothing_factor: smoothing factor of ISEF filter. Between 0-1.
    :param laplacian_kernel: kernel of laplace to use if we don't use the binary laplace
    :param laplacian_threhold: threshold value for Laplace edge detector
    :param ratio: threshold for zero crossing. Between 0-1.
    :param thinning_factor: thinning factor. Between 0-1.
    :param zc_window_size: adaptive zero crossing window size.
    :param port_output_name: result matrix
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param level:  pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :return: output image port name
    """
    if port_output_name is None:
        port_output_name = 'SHEN_CASTAN_'

        if laplacian_kernel is not None:
            port_output_name += str((laplacian_kernel.upper().replace('_XY', '')).replace('X', 'x'))

        port_output_name += '_' + port_input_name

    if is_rgb is True:
        do_grayscale_transform_job(port_input_name=port_input_name, wave_offset=wave_offset, port_output_name='GRAY_' + port_input_name)
        port_input_name = 'GRAY_' + port_input_name

    output_isef_name = 'ISEF_FILTER_F' + str(smoothing_factor).replace('.', '_') + '_' + port_input_name
    do_isef_filter_job(port_input_name=port_input_name, wave_offset=wave_offset,
                       smoothing_factor=smoothing_factor, level=level,
                       port_output_name=output_isef_name)

    if laplacian_kernel is None:
        laplace_output_name = 'LAPLACE_BINARY_ISEF_-_' + port_input_name

        do_laplacian_from_img_diff_job(port_original_input_name=port_input_name, wave_offset=wave_offset,
                                       port_smoothed_input_name=output_isef_name,
                                       do_binary=True, level=level,
                                       port_output_name=laplace_output_name)
    else:
        laplace_output_name = str((laplacian_kernel.upper().replace('_XY', '')).replace('X', 'x')) + '_THR_' + str(laplacian_threhold) +\
                              '_' + port_input_name
        do_laplace_job(port_input_name=output_isef_name, wave_offset=wave_offset, threshold_value=laplacian_threhold, level=level,
                       port_output_name=laplace_output_name, kernel=laplacian_kernel)

    zero_crossing_output_name = 'ZC_' + str(zc_window_size) + '_' + laplace_output_name
    do_zero_crossing_adaptive_window_isef_job(port_original_input_name=port_input_name, wave_offset=wave_offset,
                                              port_bli_input_name=laplace_output_name, win_size=zc_window_size,
                                              port_smoothed_input_name=output_isef_name, level=level,
                                              port_output_name=zero_crossing_output_name)

    do_threshold_hysteresis_isef_job(port_input_name=zero_crossing_output_name,
                                     ratio=ratio,  level=level,
                                     thinning_factor=thinning_factor,
                                     port_output_name=port_output_name)

    return port_output_name


# noinspection PyTypeChecker
def do_marr_hildreth_job(port_input_name: str,
                         gaussian_kernel_size: int = 0, gaussian_sigma = 0.0, threshold: float = 0.3,
                         laplacian_kernel: str = FILTERS_SECOND_ORDER.LAPLACE_1,
                         port_output_name: str = None,
                         use_precalculated_kernel: bool = False,
                         level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Marr–Hildreth algorithm is a method of detecting edges in digital images, that is, continuous curves where there are strong
    and rapid variations in image brightness. The Marr–Hildreth edge detection method is simple and operates by convolving the
    image with the Laplacian of the Gaussian function, or, as a fast approximation by difference of Gaussian.
    Then, zero crossings are detected in the filtered result to obtain the edges.
    # https://royalsocietypublishing.org/doi/abs/10.1098/rspb.1980.0020
    :param port_input_name: name of port on which we desire to apply
    :param gaussian_kernel_size: which kernel we want to use for gaussian blur
    :param gaussian_sigma: which sigma we want to use for gaussian blur
    :param laplacian_kernel: which kernel we want to use for laplace operator
    :param use_precalculated_kernel: if the laplace kernel is calculated with the gaussian blur
    :param threshold: zero crossing threshold
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    return port_output_name
    """
    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'MARR_HILDRETH_' + str((laplacian_kernel.upper().replace('_XY', '')).replace('X', 'x')) \
                           + '_S_' + str(gaussian_sigma).replace('.', '_') + '_THR_' + str(threshold).replace('.', '_') + '_' + port_input_name

    # check kernel passed
    if isinstance(laplacian_kernel, list):
        if laplacian_kernel not in custom_kernels_used:
            custom_kernels_used.append(laplacian_kernel)
        laplacian_kernel = laplacian_kernel.__str__()
    else:
        if not isinstance(laplacian_kernel, str):
            log_setup_info_to_console("CONVOLUTION JOB DIDN'T RECEIVE CORRECT KERNEL")
            return
        else:
            laplacian_kernel = laplacian_kernel.lower() + '_xy'

    output_port = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port]
    main_func_list = [input_port, wave_offset, gaussian_kernel_size, gaussian_sigma,
                      laplacian_kernel, use_precalculated_kernel, threshold, output_port]
    output_port_list = [(output_port, output_port_size, 'B', True)]

    job_name = job_name_create(action='Marr Hildreth', input_list=input_port_list, wave_offset=[wave_offset, wave_offset], level=level)

    d = create_dictionary_element(job_module='edge_second_order',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_marr_hildreth',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_dog_job(port_input_name: str,
               gaussian_kernel_size_1: int = 3, gaussian_sigma_1=1.0,
               gaussian_kernel_size_2: int = 5, gaussian_sigma_2=1.4,
               port_output_name: str = None,
               level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> str:
    """
    DoG can be seen as a single non-separable 2D convolution or the sum (difference in this case) of two separable convolutions.
    Seeing it this way, it looks like there is no computational advantage to using the DoG over the LoG.
    However, the DoG is a tunable band-pass filter, the LoG is not tunable in that same way, and should be seen as the derivative
    operator it is. The DoG also appears naturally in the scale-space setting, where the image is filtered at many scales
    (Gaussian with different sigmas), the difference between subsequent scales is a DoG.
    # https://royalsocietypublishing.org/doi/abs/10.1098/rspb.1980.0020
    :param port_input_name: name of port on which we desire to DoG on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param gaussian_kernel_size_1: which kernel we want to use for gaussian blur
    :param gaussian_sigma_1: which sigma we want to use for gaussian blur
    :param gaussian_kernel_size_2: which kernel we want to use for gaussian blur
    :param gaussian_sigma_2: which sigma we want to use for gaussian blur
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    output_name_blur_1 = 'GAUS_BLUR_K_' + str(gaussian_kernel_size_1) + '_S_' + str(gaussian_sigma_1).replace('.', '_') \
                         + '_' + port_input_name
    output_name_blur_2 = 'GAUS_BLUR_K_' + str(gaussian_kernel_size_2) + '_S_' + str(gaussian_sigma_2).replace('.', '_') \
                         + '_' + port_input_name

    if port_output_name is None:
        port_output_name = 'DoG_' + output_name_blur_1 + '_' + output_name_blur_2

    do_gaussian_blur_image_job(port_input_name=port_input_name, wave_offset=wave_offset,
                               port_output_name=output_name_blur_1, level=level, is_rgb=is_rgb,
                               kernel_size=gaussian_kernel_size_1, sigma=gaussian_sigma_1)

    do_gaussian_blur_image_job(port_input_name=port_input_name, wave_offset=wave_offset,
                               port_output_name=output_name_blur_2, level=level, is_rgb=is_rgb,
                               kernel_size=gaussian_kernel_size_2, sigma=gaussian_sigma_2)

    do_matrix_difference_job(port_input_name_1=output_name_blur_1, wave_offset_port_1=0,
                             port_input_name_2=output_name_blur_2, wave_offset_port_2=0,
                             port_output_name=port_output_name,
                             level=level, is_rgb=is_rgb)

    return port_output_name


def do_dob_job(port_input_name: str,
               kernel_size_1: int = 5, kernel_size_2: int = 3,
               port_output_name: str = None,
               level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, is_rgb: bool = False, wave_offset: int = 0) -> str:
    """
    The Difference of Box module is a filter that identifies edges. The DOB filter is similar to the LOG and DOG filters in that it is a
    two stage edge detection process. The DOB performs edge detection by performing a mean blur on an image at a specified window size.
    The resulting image is a blurred version of the source image. The module then performs another blur with a smaller window size that
    blurs the image less than previously. The final image is then calculated by replacing each pixel with the difference between the
    two blurred images and detecting when the values cross zero, i.e. negative becomes positive and vice versa. The resulting zero
    crossings will be focused at edges or areas of pixels that have some variation in their surrounding neighborhood.
    # https://ieeexplore.ieee.org/abstract/document/1671883
    :param port_input_name: name of port on which we desire to DoG on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param kernel_size_1: which kernel we want to use for gaussian blur
    :param kernel_size_2: which kernel we want to use for gaussian blur
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param is_rgb: if is colored or greyscale
    :return: output image port name
    """
    output_name_blur_1 = 'MEAN_K_' + str(kernel_size_1) + '_' + port_input_name
    output_name_blur_2 = 'MEAN_K_' + str(kernel_size_2) + '_' + port_input_name

    if port_output_name is None:
        port_output_name = 'DoB_' + output_name_blur_1 + '_' + output_name_blur_2

    do_mean_blur_job(port_input_name=port_input_name, wave_offset=wave_offset,
                     port_output_name=output_name_blur_1, level=level,
                     kernel_size=kernel_size_1, is_rgb=is_rgb)

    do_mean_blur_job(port_input_name=port_input_name, wave_offset=wave_offset,
                     port_output_name=output_name_blur_2, level=level,
                     kernel_size=kernel_size_2, is_rgb=is_rgb)

    do_matrix_difference_job(port_input_name_1=output_name_blur_1, wave_offset_port_1=0,
                             port_input_name_2=output_name_blur_2, wave_offset_port_2=0,
                             port_output_name=port_output_name,
                             level=level, is_rgb=is_rgb)

    return port_output_name


############################################################################################################################################
# line/shape detection
############################################################################################################################################


def do_hough_lines_job(port_input_name: str, vote_threshold: int,
                       distance_resolution: float = 1, angle_resolution: float = np.pi / 180,
                       start_angle_interval: float = 0, end_angle_interval: float = np.pi,
                       min_line_length: int = 0, max_line_gap: int = 0,
                       overlay: bool = False, max_number_line: int = 65535,
                       wave_offset: int = 0, port_output_name: str = None, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, ) -> None:
    """
    The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure.
    This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a
    so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform.
    The classical Hough transform was concerned with the identification of lines in the image.
    This job works only for grayscale images.
    # https://s3.cern.ch/inspire-prod-files-5/53d80b0393096ba4afe34f5b65152090
    # http://cmp.felk.cvut.cz/~matas/papers/matas-bmvc98.pdf
    :param port_input_name: name of port on which we desire to DoG on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param vote_threshold: threshold for number of votes
    :param distance_resolution: Distance resolution of the accumulator in pixels.
    :param angle_resolution: Angle resolution of the accumulator in radians.
    :param start_angle_interval: Minimum angle to check for lines. Must fall between 0 and max_theta.
    :param end_angle_interval: Maximum angle to check for lines. Must fall between min_theta and CV_PI.
    :param min_line_length: Minimum line length. Line segments shorter than that are rejected.
    :param max_line_gap: Maximum allowed gap between points on the same line to link them.
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :param overlay: overlay the lines over the image-> this will result in RGB image
    :param max_number_line: number of lines to save
    :return: None
    """

    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'HOUGH_LINE_' + port_input_name

    output_port_img = transform_port_name_lvl(name=port_output_name + '_IMG', lvl=level)
    output_port_array = transform_port_name_lvl(name=port_output_name + '_ARRAY', lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=overlay)

    input_port_list = [input_port]
    main_func_list = [input_port, wave_offset, distance_resolution, angle_resolution, vote_threshold,
                      start_angle_interval, end_angle_interval, min_line_length, max_line_gap, output_port_img, output_port_array,
                      overlay]
    output_port_list = [(output_port_img, output_port_size, 'B', True),
                        (output_port_array, '(' + str(max_number_line) + ', 1 ,4)', 'B', False)]

    job_name = job_name_create(action='Hough Lines Transform', input_list=input_port_list, wave_offset=[wave_offset, wave_offset],
                               level=level)

    d = create_dictionary_element(job_module='line_hough',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_hough',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_hough_circle_job(port_input_name: str,
                        min_dist: int, min_radius: int, max_radius: int,
                        method: str = cv2.HOUGH_GRADIENT, dp: int = 2, param1: int = 50, param2: float = 30, port_output_name: str = None,
                        overlay: bool = False, max_number_circles: int = 65535,
                        wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, ) -> None:
    """
    A Hough circle transform is an image transform that allows for circular objects to be extracted from an image, even if the circle is
    incomplete. The transform is also selective for circles, and will generally ignore elongated ellipses. The transform effectively
    searches for objects with a high degree of radial symmetry, with each degree of symmetry receiving one "vote" in the search space.
    This job works only for grayscale images.
    # https://nyuscholars.nyu.edu/en/publications/fast-contour-identification-through-efficient-hough-transform-and
    # https://www.sciencedirect.com/science/article/abs/pii/0167865588900426
    :param port_input_name: name of port on which we desire
    :param method: detection method:
                   HOUGH_GRADIENT
                   HOUGH_GRADIENT_ALT
    :param dp: Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same
               resolution as the input image. If dp=2 , the accumulator has half as big width and height.
    :param min_dist: Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles
                    may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    :param param1: First method-specific parameter. In case of #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT, it is the higher threshold of the
                   two passed to the Canny edge detector (the lower one is twice smaller).
    :param param2: Second method-specific parameter. In case of #HOUGH_GRADIENT, it is the accumulator threshold for the circle centers at
                   the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger
                   accumulator values, will be returned first. In the case of #HOUGH_GRADIENT_ALT algorithm, this is the circle
                   "perfectness" measure. The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine.
                   If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less. But then also try to
                   limit the search range [minRadius, maxRadius] to avoid many false circles.
    :param min_radius: Minimum circle radius.
    :param max_radius: Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, #HOUGH_GRADIENT returns centers without
                      finding the radius. #HOUGH_GRADIENT_ALT always computes circle radius.
    :param port_output_name: name of port output image
    :param overlay: overlay the lines over the image-> this will result in RGB image
    :param max_number_circles: number of lines to save
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: None
    """

    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'HOUGH_CIRCLE_' + port_input_name

    output_port_img = transform_port_name_lvl(name=port_output_name + '_IMG', lvl=level)
    output_port_array = transform_port_name_lvl(name=port_output_name + '_ARRAY', lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=overlay)

    input_port_list = [input_port]
    main_func_list = [input_port, wave_offset, method, dp, min_dist, param1, param2, min_radius, max_radius,
                      output_port_img, output_port_array, overlay]
    output_port_list = [(output_port_img, output_port_size, 'B', True),
                        (output_port_array, '(' + str(max_number_circles) + ', 3)', 'B', False)]

    job_name = job_name_create(action='Hough Circle Transform', input_list=input_port_list, wave_offset=[wave_offset, wave_offset],
                               level=level)

    d = create_dictionary_element(job_module='line_hough',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=None,
                                  main_func_name='main_func_hough_circle',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


############################################################################################################################################
# Image thresholding jobs
############################################################################################################################################


def do_otsu_job(port_input_name: str,
                port_output_name: str = None,
                level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0):
    """
    In the simplest form, the algorithm returns a single intensity threshold that separate pixels into two classes,
    foreground and background. This threshold is determined by minimizing intra-class intensity variance, or equivalently,
    by maximizing inter-class variance.[2] Otsu method is a one-dimensional discrete analog of Fisher's Discriminant Analysis,
    is related to Jenks optimization method, and is equivalent to a globally optimal k-means[3] performed on the intensity histogram
    Works on grayscale only
    http://webserver2.tecgraf.puc-rio.br/~mgattass/cg/trbImg/Otsu.pdf
    :param port_input_name: name of port on which we desire to apply Otsu on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'OTSU_' + port_input_name

    output_port_img = transform_port_name_lvl(name=port_output_name + '_IMG', lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)
    output_port_value = transform_port_name_lvl(name=port_output_name + '_VALUE', lvl=level)

    input_port_list = [input_port]
    main_func_list = [input_port, wave_offset, output_port_img, output_port_value]
    output_port_list = [(output_port_img, output_port_size, 'B', True), (output_port_value, '1', 'B', False)]

    job_name = job_name_create(action='Otsu Transformation', input_list=input_port_list, wave_offset=[wave_offset, wave_offset],
                               level=level)

    d = create_dictionary_element(job_module='thresholding_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func_global', init_func_param=[port_output_name],
                                  main_func_name='main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)


def do_image_threshold_job(port_input_name: str, input_value: int, input_threshold_type: str,
                           port_output_name: str = None,
                           level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    The simplest thresholding methods replace each pixel in an image with a black pixel if the image intensity or a white pixel if the
    image intensity is greater than that constant. In the example image on the right, this results in the dark tree becoming completely
    black, and the white snow becoming completely white.
    This function uses open cv for computation so please use only the specified threshold.
    :param port_input_name: name of port on which we desire to apply Otsu on
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param input_value: threshold value
    :param input_threshold_type:
                                cv2.THRESH_BINARY
                                cv2.THRESH_BINARY_INV
                                cv2.THRESH_TRUNC
                                cv2.THRESH_TOZERO
                                cv2.THRESH_TOZERO_INV
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :return: output image port name
    """
    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'THR_' + input_threshold_type.split('THRESH_')[-1] + '_' + str(input_value) + '_' \
                           + port_input_name

    output_port_img = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port]

    if isinstance(input_value, str):
        input_value = transform_port_name_lvl(name=input_value, lvl=level)
        input_port_list.append(input_value)

    main_func_list = [input_port, wave_offset, input_value, input_threshold_type, output_port_img]
    output_port_list = [(output_port_img, output_port_size, 'B', True)]

    job_name = job_name_create(action='Image threshold', input_list=input_port_list, wave_offset=[wave_offset, wave_offset], level=level,
                               THR_LVL=input_threshold_type.split('THRESH_')[-1], INPUT_VAL=str(input_value))

    d = create_dictionary_element(job_module='thresholding_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_thresholding',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


def do_image_adaptive_threshold_job(port_input_name: str,
                                    adaptive_method: str = THRESHOLD_CONFIG.THR_ADAPTIVE_MEAN_C,
                                    input_threshold_type: str = THRESHOLD_CONFIG.THR_BINARY_INV,
                                    block_size: int = 3, constant: int = 0, port_output_name: str = None,
                                    level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    In that case, we go for adaptive thresholding. In this, the algorithm calculate the threshold for a small regions of the image.
    So we get different thresholds for different regions of the same image and it gives us better results for images with varying
    illumination. This function uses open cv for computation so please use only the specified threshold.
    :param port_input_name: name of port on which we desire to apply
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param adaptive_method:
                            THRESHOLD_CONFIG.THR_ADAPTIVE_MEAN_C
                            THRESHOLD_CONFIG.THR_ADAPTIVE_GAUSS_C
    :param input_threshold_type:
                                cv2.THRESH_BINARY
                                cv2.THRESH_BINARY_INV
                                cv2.THRESH_TRUNC
                                cv2.THRESH_TOZERO
                                cv2.THRESH_TOZERO_INV
    :param block_size: size of block to consider for calculating the value
    :param constant: constant which is subtracted from the mean or weighted mean calculated.
    :param port_output_name: name of port you desire
    :param level: pyramid level to calculate at
    :return: output image port name
    """
    input_port = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'THR_ADAPTIVE_' + adaptive_method.split('ADAPTIVE_')[-1] + '_' + input_threshold_type.split('THRESH_')[-1] + \
                           '_' + str(block_size) + '_' + str(constant) + '_' + port_input_name

    output_port_img = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port]
    main_func_list = [input_port, wave_offset, adaptive_method, input_threshold_type, block_size, constant, output_port_img]
    output_port_list = [(output_port_img, output_port_size, 'B', True)]

    job_name = job_name_create(action='Image threshold', input_list=input_port_list, wave_offset=[wave_offset, wave_offset], level=level,
                               ADAPTIV_METHOD=adaptive_method.split('ADAPTIVE_')[-1], THR_LVL=input_threshold_type.split('THRESH_')[-1],
                               CONST=str(constant))

    d = create_dictionary_element(job_module='thresholding_image',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_func_adaptive_thresholding',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


############################################################################################################################################
# Skeletonization/thinning jobs
############################################################################################################################################


def do_thinning_guo_hall_image_job(port_input_name: str,
                                   port_output_name: str = None,
                                   level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Guo - Hall explained in "Parallel thinning with two sub-iteration algorithms" by Zicheng Guo and Richard Hall
    # https://dl.acm.org/doi/abs/10.1145/62065.62074
    :param port_input_name: name of input port
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :param port_output_name: name of output port
    :param level: pyramid level to calculate at
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'THINNING_GUO_HALL_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Guo-Hall thinning', input_list=input_port_list, wave_offset=[wave_offset, wave_offset], level=level)

    d = create_dictionary_element(job_module='thinning',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  max_wave=wave_offset,
                                  init_func_name='init_func', init_func_param=None,
                                  main_func_name='main_guo_hall_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


############################################################################################################################################
# Line/edge connectivity jobs
############################################################################################################################################


def do_edge_label_job(port_input_name: str,
                      port_output_name: str = None, port_output_label_name: str = None, connectivity: int = 8,
                      wave_offset: int = 0, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0) -> str:
    """
    4-connected pixels are neighbors to every pixel that touches one of their edges or corners.
    These pixels are connected horizontally, vertically.
    8-connected pixels are neighbors to every pixel that touches one of their edges or corners.
    These pixels are connected horizontally, vertically, and diagonally.
    # https://books.google.ro/books?hl=en&lr=&id=ANiNDAAAQBAJ&oi=fnd&pg=PP1&dq=Digital+Picture+Processing,+Academic+Press&ots=vgFslrrBQi&sig=Y1EHfka_nxhk4rbO0JKlnqyuzPQ&redir_esc=y#v=onepage&q=Digital%20Picture%20Processing%2C%20Academic%20Press&f=false
    :param port_input_name: name of input port
    :param connectivity: 8 or 4 for 8-way or 4-way connectivity respectively
    :param port_output_name: name of output port
    :param port_output_label_name: name of output port label
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: None
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_output_name is None:
        port_output_name = 'EDGE_LABELED_RGB_' + str(connectivity) + '_' + port_input_name
        port_output_label_name = 'EDGE_LABELED_' + str(connectivity) + '_' + port_input_name

    output_port_name_rgb = transform_port_name_lvl(name=port_output_name, lvl=level)
    output_port_size_rgb = transform_port_size_lvl(lvl=level, rgb=True)

    output_port_name = transform_port_name_lvl(name=port_output_label_name, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, connectivity, output_port_name_rgb, output_port_name]
    output_port_list = [(output_port_name_rgb, output_port_size_rgb, 'B', True), (output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Connected Line Labeling', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='line_connectivity',
                                  job_name=job_name,
                                  max_wave=wave_offset,
                                  input_ports=input_port_list,
                                  init_func_name='init_edge_label', init_func_param=input_port_list,
                                  main_func_name='create_edge_label_map',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_output_name


############################################################################################################################################
# U-Net jobs
############################################################################################################################################


def do_u_net_edge(port_input_name: str, location_model: str = 'unet_edge',
                  port_name_output: str = None,
                  is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    U-Net model trained to detect edges. Can be applied only in current wave.
    :param port_input_name: name of input port
    :param location_model: model name
    :param port_name_output: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_name_output is None:
        port_name_output = 'EDGE_UNET_' + port_input_name

    output_port_name = transform_port_name_lvl(name=port_name_output, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, output_port_name]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    job_name = job_name_create(action='Edge U-Net', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module='u_net',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func', init_func_param=[location_model],
                                  main_func_name='main_run_unet_edge_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_name_output


############################################################################################################################################
# Semseg jobs
############################################################################################################################################
def do_u_net_semseg(port_input_name: str, number_of_classes: int, save_img_augmentation: bool = False,
                    save_overlay: bool = False, save_legend_in_image: bool = False,
                    list_colors_to_use=None, list_class_name=None,
                    port_name_output: str = None,
                    is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    TODO add details
    https://arxiv.org/abs/1505.04597
    :param port_input_name: name of input port
    :param number_of_classes: number of classes trained in the network
    :param save_img_augmentation: if we want to save the augmented image to.
    :param save_overlay: if we want to save overlay of the augmented and raw image to.
    :param list_colors_to_use: list of exact colors per class to use
    :param save_legend_in_image: if we want to save the legend in the image
    :param list_class_name: list of class names to use
    :param port_name_output: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """

    if list_class_name is None:
        list_class_name = []
    if list_colors_to_use is None:
        list_colors_to_use = []
    return do_semseg_base_job(port_input_name=port_input_name, number_of_classes=number_of_classes, model='unet',
                              save_img_augmentation=save_img_augmentation, save_overlay=save_overlay,
                              save_legend_in_image=save_legend_in_image, list_colors_to_use=list_colors_to_use,
                              list_class_name=list_class_name, port_name_output=port_name_output,
                              is_rgb=is_rgb, level=level, wave_offset=wave_offset)


def do_vgg_u_net_semseg(port_input_name: str, number_of_classes: int, save_img_augmentation: bool = False,
                        save_overlay: bool = False, save_legend_in_image: bool = False,
                        list_colors_to_use=None, list_class_name=None,
                        port_name_output: str = None,
                        is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    TODO add details
    https://arxiv.org/abs/1411.4038
    :param port_input_name: name of input port
    :param number_of_classes: number of classes trained in the network
    :param save_img_augmentation: if we want to save the augmented image to.
    :param save_overlay: if we want to save overlay of the augmented and raw image to.
    :param list_colors_to_use: list of exact colors per class to use
    :param save_legend_in_image: if we want to save the legend in the image
    :param list_class_name: list of class names to use
    :param port_name_output: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """

    if list_class_name is None:
        list_class_name = []
    if list_colors_to_use is None:
        list_colors_to_use = []
    return do_semseg_base_job(port_input_name=port_input_name, number_of_classes=number_of_classes, model='vgg_unet',
                              save_img_augmentation=save_img_augmentation, save_overlay=save_overlay,
                              save_legend_in_image=save_legend_in_image, list_colors_to_use=list_colors_to_use,
                              list_class_name=list_class_name, port_name_output=port_name_output,
                              is_rgb=is_rgb, level=level, wave_offset=wave_offset)


def do_resnet50_unet_semseg(port_input_name: str, number_of_classes: int, save_img_augmentation: bool = False,
                            save_overlay: bool = False, save_legend_in_image: bool = False,
                            list_colors_to_use=None, list_class_name=None,
                            port_name_output: str = None,
                            is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    TODO add details
    Paper:     TODO add paper
    :param port_input_name: name of input port
    :param number_of_classes: number of classes trained in the network
    :param save_img_augmentation: if we want to save the augmented image to.
    :param save_overlay: if we want to save overlay of the augmented and raw image to.
    :param list_colors_to_use: list of exact colors per class to use
    :param save_legend_in_image: if we want to save the legend in the image
    :param list_class_name: list of class names to use
    :param port_name_output: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """

    if list_class_name is None:
        list_class_name = []
    if list_colors_to_use is None:
        list_colors_to_use = []
    return do_semseg_base_job(port_input_name=port_input_name, number_of_classes=number_of_classes, model='resnet50_unet',
                              save_img_augmentation=save_img_augmentation, save_overlay=save_overlay,
                              save_legend_in_image=save_legend_in_image, list_colors_to_use=list_colors_to_use,
                              list_class_name=list_class_name, port_name_output=port_name_output,
                              is_rgb=is_rgb, level=level, wave_offset=wave_offset)


def do_unet_mini_semseg(port_input_name: str, number_of_classes: int, save_img_augmentation: bool = False,
                        save_overlay: bool = False, save_legend_in_image: bool = False,
                        list_colors_to_use=None, list_class_name=None,
                        port_name_output: str = None,
                        is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    TODO add details
    Paper:     TODO add paper
    :param port_input_name: name of input port
    :param number_of_classes: number of classes trained in the network
    :param save_img_augmentation: if we want to save the augmented image to.
    :param save_overlay: if we want to save overlay of the augmented and raw image to.
    :param list_colors_to_use: list of exact colors per class to use
    :param save_legend_in_image: if we want to save the legend in the image
    :param list_class_name: list of class names to use
    :param port_name_output: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """

    if list_class_name is None:
        list_class_name = []
    if list_colors_to_use is None:
        list_colors_to_use = []
    return do_semseg_base_job(port_input_name=port_input_name, number_of_classes=number_of_classes, model='unet_mini',
                              save_img_augmentation=save_img_augmentation, save_overlay=save_overlay,
                              save_legend_in_image=save_legend_in_image, list_colors_to_use=list_colors_to_use,
                              list_class_name=list_class_name, port_name_output=port_name_output,
                              is_rgb=is_rgb, level=level, wave_offset=wave_offset)


def do_mobilenet_unet_semseg(port_input_name: str, number_of_classes: int, save_img_augmentation: bool = False,
                             save_overlay: bool = False, save_legend_in_image: bool = False,
                             list_colors_to_use=None, list_class_name=None,
                             port_name_output: str = None,
                             is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    TODO add details
    Paper:     TODO add paper
    :param port_input_name: name of input port
    :param number_of_classes: number of classes trained in the network
    :param save_img_augmentation: if we want to save the augmented image to.
    :param save_overlay: if we want to save overlay of the augmented and raw image to.
    :param list_colors_to_use: list of exact colors per class to use
    :param save_legend_in_image: if we want to save the legend in the image
    :param list_class_name: list of class names to use
    :param port_name_output: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """

    if list_class_name is None:
        list_class_name = []
    if list_colors_to_use is None:
        list_colors_to_use = []
    return do_semseg_base_job(port_input_name=port_input_name, number_of_classes=number_of_classes, model='mobilenet_unet',
                              save_img_augmentation=save_img_augmentation, save_overlay=save_overlay,
                              save_legend_in_image=save_legend_in_image, list_colors_to_use=list_colors_to_use,
                              list_class_name=list_class_name, port_name_output=port_name_output,
                              is_rgb=is_rgb, level=level, wave_offset=wave_offset)


# noinspection PyUnboundLocalVariable
def do_semseg_base_job(port_input_name: str, number_of_classes: int, model: str,
                       save_img_augmentation=False,
                       save_overlay: bool = False, save_legend_in_image: bool = False,
                       list_colors_to_use=None, list_class_name=None,
                       port_name_output: str = None,
                       is_rgb: bool = False, level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    TODO add details
    :param port_input_name: name of input port
    :param number_of_classes: number of classes trained in the network
    :param model: model to use for training
                    ["fcn_8", "fcn_32", "fcn_8_vgg", "fcn_32_vgg", "fcn_8_resnet50", "fcn_32_resnet50", "fcn_8_mobilenet",
                     "fcn_32_mobilenet", "pspnet", "vgg_pspnet", "resnet50_pspnet", "vgg_pspnet", "resnet50_pspnet", "pspnet_50",
                     "pspnet_101", "unet_mini", "unet", "vgg_unet", "resnet50_unet", "mobilenet_unet", "segnet", "vgg_segnet",
                     "resnet50_segnet", "mobilenet_segnet" ]
    :param save_img_augmentation: if we want to save the augmented image to.
    :param save_overlay: if we want to save overlay of the augmented and raw image to.
    :param list_colors_to_use: list of exact colors per class to use
    :param save_legend_in_image: if we want to save the legend in the image
    :param list_class_name: list of class names to use
    :param port_name_output: name of output port
    :param level: pyramid level to calculate at
    :param is_rgb: if the output ports is rgb, 3 channels
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """
    if list_colors_to_use is None:
        list_colors_to_use = []
    if list_class_name is None:
        list_class_name = []
    model_list = ["fcn_8", "fcn_32", "fcn_8_vgg", "fcn_32_vgg", "fcn_8_resnet50", "fcn_32_resnet50", "fcn_8_mobilenet", "fcn_32_mobilenet",
                  "pspnet", "vgg_pspnet", "resnet50_pspnet", "vgg_pspnet", "resnet50_pspnet", "pspnet_50", "pspnet_101", "unet_mini",
                  "unet", "vgg_unet", "resnet50_unet", "mobilenet_unet", "segnet", "vgg_segnet", "resnet50_segnet", "mobilenet_segnet"]

    model_name = None

    if model in model_list:
        model_name = model

    location_model = 'MachineLearning\\model_weights\\' + model_name

    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_name_output is None:
        port_name_output = 'SEMSEG_' + model_name.upper() + '_' + port_input_name

    if save_img_augmentation is True:
        port_name_output_overlay = 'OVERLAY_' + port_name_output

    output_port_name = transform_port_name_lvl(name=port_name_output, lvl=level)
    output_port_size = transform_port_size_lvl(lvl=level, rgb=is_rgb)

    if save_img_augmentation is True:
        output_overlay_port_name = transform_port_name_lvl(name=port_name_output_overlay, lvl=level)
        output_overlay_port_size = transform_port_size_lvl(lvl=level, rgb=True)

    if save_legend_in_image is True and list_class_name == []:
        log_to_console('PLEASE PASS LIST OF CLASS NAMES FOR LEGEND')
        save_legend_in_image = False

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, output_port_name, save_img_augmentation, number_of_classes, save_overlay,
                      save_legend_in_image, list_colors_to_use, list_class_name, model_name, None]
    output_port_list = [(output_port_name, output_port_size, 'B', True)]

    if save_img_augmentation is True:
        output_port_list.append((output_overlay_port_name, output_overlay_port_size, 'B', True))
        main_func_list[-1] = output_overlay_port_name

    job_name = job_name_create(action='SemSeg' + model_name.replace('_', ' '), input_list=input_port_list, wave_offset=[wave_offset],
                               level=level)

    d = create_dictionary_element(job_module='semseg',
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_semseg_keras_repo', init_func_param=[location_model, model_name, output_port_size],
                                  main_func_name='main_func_semseg_keras_repo',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_name_output


if __name__ == "__main__":
    pass
