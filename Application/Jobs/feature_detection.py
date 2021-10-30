# noinspection PyPackageRequirements
from typing import Tuple

# Do not delete used indirectly
# noinspection PyUnresolvedReferences
from Application.Frame import transferJobPorts
from Application.Frame.global_variables import JobInitStateReturn, global_var_handler
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_error_to_console, log_to_console
from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL, APPL_SAVE_LOCATION
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file

from Application.Utils.misc import save_keypoint_to_array
import cv2
import os
import numpy as np


############################################################################################################################################
# Init functions
############################################################################################################################################


def init_func_sift() -> JobInitStateReturn:
    """
    Init function for sift algorithm
    :return: INIT or NOT_INIT state for the job
    """
    return JobInitStateReturn(True)

############################################################################################################################################
# Main functions
############################################################################################################################################

def main_func_shift(param_list: list = None) -> bool:
    """
    Main function for SIFT calculation job.
    :param param_list: Param needed to respect the following list:
                       [input_port_name, input_port_wave, number_of_features, number_of_octaves, contrast_threshold, edge_threshold,
                        sigma_gaussian, port_name_mask, port_output_keypoints,  port_output_descriptors, port_output_img]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_NR_FEATURES = 2
    # noinspection PyPep8Naming
    PORT_IN_NR_OCTAVE_LAYERS = 3
    # noinspection PyPep8Naming
    PORT_IN_CONTRAST_THR = 4
    # noinspection PyPep8Naming
    PORT_IN_EDGE_THR = 5
    # noinspection PyPep8Naming
    PORT_IN_SIGMA = 6
    # noinspection PyPep8Naming
    PORT_IN_MASK = 7
    # noinspection PyPep8Naming
    PORT_OUT_KEYPOINTS = 8
    # noinspection PyPep8Naming
    PORT_OUT_DESCRIPTORS = 9
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 10

    # verify that the number of parameters are OK.
    if len(param_list) != 11:
        log_error_to_console("SIFT JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        p_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        # set mask port to input image if none
        if param_list[PORT_IN_MASK] is not None:
            p_in_mask = get_port_from_wave(name=param_list[PORT_IN_MASK], wave_offset=param_list[PORT_IN_WAVE])
        else:
            p_in_mask = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        # get output port
        p_out_keypoints = get_port_from_wave(name=param_list[PORT_OUT_KEYPOINTS])
        p_out_des = get_port_from_wave(name=param_list[PORT_OUT_DESCRIPTORS])
        p_out_img = get_port_from_wave(name=param_list[PORT_OUT_IMG])

        # check if port's you want to use are valid
        if p_in.is_valid() is True:
            try:
                sift_obj = cv2.SIFT_create(nfeatures=param_list[PORT_IN_NR_FEATURES], nOctaveLayers=param_list[PORT_IN_NR_OCTAVE_LAYERS],
                                           contrastThreshold=param_list[PORT_IN_CONTRAST_THR], edgeThreshold=param_list[PORT_IN_EDGE_THR],
                                           sigma=param_list[PORT_IN_SIGMA])

                kp, des = sift_obj.detectAndCompute(image=p_in.arr.copy(), mask=p_in_mask.arr)
                # image of features
                tmp = cv2.drawKeypoints(image=p_in.arr.copy(), keypoints=kp, outImage=p_out_img.arr.copy())
                p_out_img.arr[:] = tmp[:]
                p_out_img.set_valid()
                # save KeyPoints to port
                for idx in range(min(len(des), param_list[PORT_IN_NR_FEATURES])):
                    p_out_des.arr[idx][:] = des[idx]
                p_out_des.set_valid()
                # save descriptors to port
                for t in range(min(len(kp), param_list[PORT_IN_NR_FEATURES])):
                    tmp = save_keypoint_to_array(kp[t])
                    p_out_keypoints.arr[t][:] = tmp
                p_out_keypoints.set_valid()

            except BaseException as error:
                log_error_to_console("SIFT JOB NOK: ", str(error))
                pass
        else:
            return False

        return True


def main_func_a_kaze(param_list: list = None) -> bool:
    """
    Main function for A-KAZE calculation job.
    :param param_list: Param needed to respect the following list:
                       [input_port_name, input_port_wave, number_of_features, number_of_octaves, contrast_threshold, edge_threshold,
                        sigma_gaussian, port_name_mask, port_output_keypoints,  port_output_descriptors, port_output_img]
    :return: True if the job executed OK.
    """
    # noinspection PyPep8Naming
    PORT_IN_POS = 0
    # noinspection PyPep8Naming
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_NR_FEATURES = 2
    # noinspection PyPep8Naming
    PORT_IN_DESC_TYPE = 3
    # noinspection PyPep8Naming
    PORT_IN_DESC_SIZE = 4
    # noinspection PyPep8Naming
    PORT_IN_DESC_CHANNELS = 5
    # noinspection PyPep8Naming
    PORT_IN_THR = 6
    # noinspection PyPep8Naming
    PORT_IN_N_OCTAVES = 7
    # noinspection PyPep8Naming
    PORT_IN_N_OCTAVES_LAYERS = 8
    # noinspection PyPep8Naming
    PORT_IN_DIFFUSIVITY = 9
    # noinspection PyPep8Naming
    PORT_IN_MASK = 10
    # noinspection PyPep8Naming
    PORT_IN_SAVE_NPY = 11
    # noinspection PyPep8Naming
    PORT_IN_SAVE_TXT = 12
    # noinspection PyPep8Naming
    PORT_OUT_KEYPOINTS = 13
    # noinspection PyPep8Naming
    PORT_OUT_DESCRIPTORS = 14
    # noinspection PyPep8Naming
    PORT_OUT_IMG = 15

    # verify that the number of parameters are OK.
    if len(param_list) != 16:
        log_error_to_console("A-KAZE JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        p_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        # set mask port to input image if none
        if param_list[PORT_IN_MASK] is not None:
            p_in_mask = get_port_from_wave(name=param_list[PORT_IN_MASK], wave_offset=param_list[PORT_IN_WAVE])
        else:
            p_in_mask = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        # get output port
        p_out_keypoints = get_port_from_wave(name=param_list[PORT_OUT_KEYPOINTS])
        p_out_des = get_port_from_wave(name=param_list[PORT_OUT_DESCRIPTORS])
        p_out_img = get_port_from_wave(name=param_list[PORT_OUT_IMG])

        # check if port's you want to use are valid
        if p_in.is_valid() is True:
            try:
                a_kaze_obj = cv2.AKAZE_create(descriptor_type=param_list[PORT_IN_DESC_TYPE], descriptor_size=param_list[PORT_IN_DESC_SIZE],
                                              descriptor_channels=param_list[PORT_IN_DESC_CHANNELS], threshold=param_list[PORT_IN_THR],
                                              nOctaves=param_list[PORT_IN_N_OCTAVES], nOctaveLayers=param_list[PORT_IN_N_OCTAVES_LAYERS],
                                              diffusivity=param_list[PORT_IN_DIFFUSIVITY])
                
                kp, des = a_kaze_obj.detectAndCompute(image=p_in.arr.copy(), mask=p_in_mask.arr)
                # image of features
                tmp = cv2.drawKeypoints(image=p_in.arr.copy(), keypoints=kp, outImage=p_out_img.arr.copy())
                # p_out_img.arr[:] = tmp[:]
                p_out_img.arr[:] = tmp[:]
                p_out_img.set_valid()
                # save KeyPoints to port
                # for idx in range(min(len(des), param_list[PORT_IN_NR_FEATURES])):
                #     p_out_des.arr[idx][:] = des[idx]

                p_out_des.arr = np.float32(des)

                file_to_save = os.path.join(APPL_SAVE_LOCATION, p_out_des.name)

                if param_list[PORT_IN_SAVE_NPY]:
                    if not os.path.exists(file_to_save):
                        os.makedirs(file_to_save)

                    location_np = os.path.join(file_to_save, global_var_handler.PICT_NAME[:-4])
                    np.save(location_np, p_out_des.arr)

                if param_list[PORT_IN_SAVE_TXT]:
                    if not os.path.exists(file_to_save):
                        os.makedirs(file_to_save)

                    location_np = os.path.join(file_to_save, global_var_handler.PICT_NAME[:-4] + '.txt')
                    np.savetxt(location_np, p_out_des.arr)

                p_out_des.set_valid()
                # save descriptors to port
                for t in range(min(len(kp), param_list[PORT_IN_NR_FEATURES])):
                    tmp = save_keypoint_to_array(kp[t])
                    p_out_keypoints.arr[t][:] = tmp

                file_to_save = os.path.join(APPL_SAVE_LOCATION, p_out_keypoints.name)

                if param_list[PORT_IN_SAVE_TXT]:
                    if not os.path.exists(file_to_save):
                        os.makedirs(file_to_save)

                    location_np = os.path.join(file_to_save, global_var_handler.PICT_NAME.split('.')[0])
                    np.save(location_np, p_out_keypoints.arr)

                if param_list[PORT_IN_SAVE_NPY]:
                    if not os.path.exists(file_to_save):
                        os.makedirs(file_to_save)

                    location_np = os.path.join(file_to_save, global_var_handler.PICT_NAME.split('.')[0] + '.txt')
                    np.savetxt(location_np, p_out_keypoints.arr)

                p_out_keypoints.set_valid()



            except BaseException as error:
                log_error_to_console("A_KAZE JOB NOK: ", str(error))
                pass
        else:
            return False

        return True

############################################################################################################################################
# Job create functions
############################################################################################################################################


def do_sift_job(port_input_name: str,
                number_features: int = 512, number_octaves: int = 3, contrast_threshold: float = 0.04, edge_threshold: int = 10,
                gaussian_sigma: float = 1.6, mask_port_name: str = None,
                port_kp_output: str = None, port_des_output: str = None, port_img_output: str = None,
                level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> Tuple[str, str, str]:
    """
    The scale-invariant feature transform (SIFT) is a feature detection algorithm in computer vision to detect and describe local features
    in images.SIFT keypoints of objects are first extracted from a set of reference images[1] and stored in a database. An object is
    recognized in a new image by individually comparing each feature from the new image to this database and finding candidate matching
    features based on Euclidean distance of their feature vectors. From the full set of matches, subsets of keypoints that agree on the
    object and its location, scale, and orientation in the new image are identified to filter out good matches. The determination of
    consistent clusters is performed rapidly by using an efficient hash table implementation of the generalised Hough transform.
    Each cluster of 3 or more features that agree on an object and its pose is then subject to further detailed model verification and
    subsequently outliers are discarded. Finally the probability that a particular set of features indicates the presence of an object is
    computed, given the accuracy of fit and number of probable false matches. Object matches that pass all these tests can be identified
    as correct with high confidence
    https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf
    :param port_input_name: name of input port
    :param number_features: number of features to save, they are ranked by score
    :param number_octaves: The number of layers in each octave.
    :param contrast_threshold: The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
                               The larger the threshold, the less features are produced by the detector.
    :param edge_threshold: The threshold used to filter out edge-like features. Note that the its meaning is different from the
                           contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out.
    :param gaussian_sigma: The sigma of the Gaussian applied to the input image at the octave #0.
    :param mask_port_name: Masks for each input image specifying where to look for keypoints
    :param port_kp_output: Output port name for KeyPoints list
    :param port_des_output: Output port name for descriptor list
    :param port_img_output: Output port name for image
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """

    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_kp_output is None:
        port_kp_output = 'SIFT_KP_NF_{nf}_NO_{no}_CT_{ct}_ET_{et}_G_{g}_{INPUT}'.format(nf=number_features, no=number_octaves,
                                                                                        ct=contrast_threshold.__str__().replace('.', '_'),
                                                                                        et=edge_threshold, g=gaussian_sigma.__str__().replace('.', '_'),
                                                                                        INPUT=port_input_name)
        if mask_port_name is not None:
            port_kp_output += '_MASKED_BY_' + mask_port_name

    if port_des_output is None:
        port_des_output = 'SIFT_DES_NF_{nf}_NO_{no}_CT_{ct}_ET_{et}_G_{g}_{INPUT}'.format(nf=number_features, no=number_octaves,
                                                                                          ct=contrast_threshold.__str__().replace('.', '_'),
                                                                                          et=edge_threshold, g=gaussian_sigma.__str__().replace('.', '_'),
                                                                                          INPUT=port_input_name)
        if mask_port_name is not None:
            port_des_output += '_MASKED_BY_' + mask_port_name

    if port_img_output is None:
        port_img_output = 'SIFT_IMG_NF_{nf}_NO_{no}_CT_{ct}_ET_{et}_G_{g}_{INPUT}'.format(nf=number_features, no=number_octaves,
                                                                                          ct=contrast_threshold.__str__().replace('.', '_'),
                                                                                          et=edge_threshold, g=gaussian_sigma.__str__().replace('.', '_'),
                                                                                          INPUT=port_input_name)
        if mask_port_name is not None:
            port_img_output += '_MASKED_BY_' + mask_port_name

    port_kp_output_name = transform_port_name_lvl(name=port_kp_output, lvl=level)
    port_kp_output_name_size = '({nr_kp}, {size_kp})'.format(nr_kp=number_features, size_kp=7)

    port_des_output_name = transform_port_name_lvl(name=port_des_output, lvl=level)
    port_des_output_name_size = '({nr_kp}, 128)'.format(nr_kp=number_features)

    port_img_output_name = transform_port_name_lvl(name=port_img_output, lvl=level)
    port_img_output_name_size = transform_port_size_lvl(lvl=level, rgb=True)

    if mask_port_name is not None:
        mask_port_name = transform_port_name_lvl(name=mask_port_name, lvl=level)

    input_port_list = [input_port_name]

    main_func_list = [input_port_name, wave_offset, number_features, number_octaves, contrast_threshold, edge_threshold, gaussian_sigma,
                      mask_port_name, port_kp_output_name, port_des_output_name, port_img_output_name]

    output_port_list = [(port_kp_output_name, port_kp_output_name_size, 'H', False),
                        (port_des_output_name, port_des_output_name_size, 'H', False),
                        (port_img_output_name, port_img_output_name_size, 'B', True)]

    job_name = job_name_create(action='SIFT', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_sift', init_func_param=None,
                                  main_func_name='main_func_shift',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_kp_output, port_des_output, port_img_output


def do_a_kaze_job(port_input_name: str, number_features: int = 1024,
                  descriptor_type: int = cv2.AKAZE_DESCRIPTOR_KAZE, descriptor_size: int = 0, descriptor_channels: int = 3,
                  threshold: float = 0.001, nr_octaves: int = 4, nr_octave_layers: int = 4, diffusivity: int = cv2.KAZE_DIFF_PM_G1,
                  save_to_text: bool = True, save_to_npy: bool = True,
                  mask_port_name: str = None,
                  port_kp_output: str = None, port_des_output: str = None, port_img_output: str = None,
                  level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> Tuple[str, str, str]:
    """
    Accelerated-KAZE (A-KAZE) feature detector and descriptor uses nonlinear scale spaces to extract corresponding features of images.
    A-KAZE is developed using KAZE algorithm and embedding mathematical Fast Explicit Diffusion (FED) in pyramidal structure to speed up the nonlinear scale space computation.
    A-KAZE is comprised of three steps: a nonlinear scale space building with FED, feature detection, and feature description.
    # http://www.bmva.org/bmvc/2013/Papers/paper0013/paper0013.pdf
    :param port_input_name: name of input port
    :param number_features: number of features to save, they are ranked by score
    :param descriptor_type: Type of the extracted descriptor
           cv2.AKAZE_DESCRIPTOR_KAZE, cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT, cv2.AKAZE_DESCRIPTOR_MLDB, cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT
    :param descriptor_size: Size of the descriptor in bits. 0 -> Full size
    :param descriptor_channels: Number of channels in the descriptor (1, 2, 3)
    :param threshold: Detector response threshold to accept point
    :param nr_octaves: Maximum octave evolution of the image
    :param nr_octave_layers: Default number of sublevels per scale level
    :param diffusivity: Diffusivity type.
           cv2.KAZE_DIFF_PM_G1, cv2.KAZE_DIFF_PM_G2, cv2.KAZE_DIFF_CHARBONNIER, cv2.KAZE_DIFF_WEICKERT
    :param mask_port_name: Masks for each input image specifying where to look for keypoints
    :param save_to_text: if we want to save BOW clusters to txt files
    :param save_to_npy: if we want to save BOW clusters to npy files
    :param port_kp_output: Output port name for KeyPoints list
    :param port_des_output: Output port name for descriptor list
    :param port_img_output: Output port name for image
    :param level: pyramid level to calculate at
    :param wave_offset: port wave offset. If 0 it is in current wave.
    :return: output image port name
    """

    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_kp_output is None:
        port_kp_output = 'A_KAZE_KP_DT_{dt}_DS_{ds}_THR_{thr}_NO_{no}_NOL_{nol}_D_{d}_{INPUT}'.format(dt=descriptor_type.__str__(), ds=descriptor_size.__str__(),
                                                                                                   thr=threshold.__str__().replace('.', '_'), d=diffusivity.__str__(),
                                                                                                   no=nr_octaves.__str__(), nol=nr_octave_layers.__str__().replace('.', '_'),
                                                                                                   INPUT=port_input_name)
        if mask_port_name is not None:
            port_kp_output += '_MASKED_BY_' + mask_port_name

    if port_des_output is None:
        port_des_output = 'A_KAZE_DES_DT_{dt}_DS_{ds}_THR_{thr}_NO_{no}_NOL_{nol}_D_{d}_{INPUT}'.format(dt=descriptor_type.__str__(), ds=descriptor_size.__str__(),
                                                                                                        thr=threshold.__str__().replace('.', '_'), d=diffusivity.__str__(),
                                                                                                        no=nr_octaves.__str__(), nol=nr_octave_layers.__str__().replace('.', '_'),
                                                                                                        INPUT=port_input_name)
        if mask_port_name is not None:
            port_des_output += '_MASKED_BY_' + mask_port_name

    if port_img_output is None:
        port_img_output = 'A_KAZE_IMG_DT_{dt}_DS_{ds}_THR_{thr}_NO_{no}_NOL_{nol}_D_{d}_{INPUT}'.format(dt=descriptor_type.__str__(), ds=descriptor_size.__str__(),
                                                                                                        thr=threshold.__str__().replace('.', '_'), d=diffusivity.__str__(),
                                                                                                        no=nr_octaves.__str__(), nol=nr_octave_layers.__str__().replace('.', '_'),
                                                                                                        INPUT=port_input_name)
        if mask_port_name is not None:
            port_img_output += '_MASKED_BY_' + mask_port_name

    port_kp_output_name = transform_port_name_lvl(name=port_kp_output, lvl=level)
    port_kp_output_name_size = '({nr_kp}, {size_kp})'.format(nr_kp=number_features, size_kp=7)

    port_des_output_name = transform_port_name_lvl(name=port_des_output, lvl=level)
    port_des_output_name_size = '({nr_kp}, {size_desc})'.format(nr_kp=number_features, size_desc=64)

    port_img_output_name = transform_port_name_lvl(name=port_img_output, lvl=level)
    port_img_output_name_size = transform_port_size_lvl(lvl=level, rgb=True)

    if mask_port_name is not None:
        mask_port_name = transform_port_name_lvl(name=mask_port_name, lvl=level)

    input_port_list = [input_port_name]

    main_func_list = [input_port_name, wave_offset, number_features, descriptor_type, descriptor_size, descriptor_channels,
                      threshold, nr_octaves, nr_octave_layers, diffusivity,
                      mask_port_name,  save_to_text, save_to_npy,
                      port_kp_output_name, port_des_output_name, port_img_output_name]

    output_port_list = [(port_kp_output_name, port_kp_output_name_size, 'f', False),
                        (port_des_output_name, port_des_output_name_size, 'f', False),
                        (port_img_output_name, port_img_output_name_size, 'B', True)]

    job_name = job_name_create(action='A-KAZE', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_sift', init_func_param=None,
                                  main_func_name='main_func_a_kaze',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_kp_output, port_des_output, port_img_output


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
