# import what you need
import cv2
import numpy as np
import skimage.feature
import skimage.measure
import matplotlib.pyplot as plt

from Application.Frame.global_variables import JobInitStateReturn
from Application.Frame.transferJobPorts import get_port_from_wave
from Utils.log_handler import log_to_console, log_to_file, log_error_to_console

from Application.Config.create_config import jobs_dict, create_dictionary_element
from config_main import PYRAMID_LEVEL
from Application.Config.util import transform_port_name_lvl, transform_port_size_lvl, job_name_create, get_module_name_from_file


"""
Module handles grey-level co-occurrence matrix jobs for the APPL block.
"""

############################################################################################################################################
# Internal functions
############################################################################################################################################

############################################################################################################################################
# Init functions
############################################################################################################################################

# define a init function, function that will be executed at the begging of the wave
def init_func_global(param_list: list = None) -> JobInitStateReturn:
    """
    Init function for the job.
    Remember this function is called before the framework gets pictures.
    :return: INIT or NOT_INIT state for the job
    """
    # noinspection PyPep8Naming
    PORT_IN_CON = 0
    # noinspection PyPep8Naming
    PORT_IN_DIS = 1
    # noinspection PyPep8Naming
    PORT_IN_HOM = 2
    # noinspection PyPep8Naming
    PORT_IN_ASM = 3
    # noinspection PyPep8Naming
    PORT_IN_ENE = 4
    # noinspection PyPep8Naming
    PORT_IN_COR = 5
    # noinspection PyPep8Naming
    PORT_IN_ENT = 6
    # noinspection PyPep8Naming
    PORT_IN_INPUT = 7

    if param_list[PORT_IN_CON]:
        log_to_file('GLCM CONTRAST ' + param_list[PORT_IN_INPUT])
    if param_list[PORT_IN_DIS]:
        log_to_file('GLCM DISSIMILARITY ' + param_list[PORT_IN_INPUT])
    if param_list[PORT_IN_HOM]:
        log_to_file('GLCM HOMOGENEITY ' + param_list[PORT_IN_INPUT])
    if param_list[PORT_IN_ASM]:
        log_to_file('GLCM ASM ' + param_list[PORT_IN_INPUT])
    if param_list[PORT_IN_ENE]:
        log_to_file('GLCM ENERGY ' + param_list[PORT_IN_INPUT])
    if param_list[PORT_IN_COR]:
        log_to_file('GLCM CORRELATION ' + param_list[PORT_IN_INPUT])
    if param_list[PORT_IN_ENT]:
        log_to_file('GLCM ENTROPY ' + param_list[PORT_IN_INPUT])

    return JobInitStateReturn(True)


############################################################################################################################################
# Main functions
############################################################################################################################################

# define a main function, function that will be executed at the begging of the wave
def main_func(param_list: list = None) -> bool:
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
    PORT_IN_WAVE = 1
    # noinspection PyPep8Naming
    PORT_IN_DISTANCE = 2
    # noinspection PyPep8Naming
    PORT_IN_ANGLE = 3
    # noinspection PyPep8Naming
    PORT_IN_MATRIX_LVL = 4
    # noinspection PyPep8Naming
    PORT_IN_MATRIX_SYMMETRY = 5
    # noinspection PyPep8Naming
    PORT_IN_MATRIX_NORMED = 6
    # noinspection PyPep8Naming
    PORT_IN_CON = 7
    # noinspection PyPep8Naming
    PORT_IN_DIS = 8
    # noinspection PyPep8Naming
    PORT_IN_HOM = 9
    # noinspection PyPep8Naming
    PORT_IN_ASM = 10
    # noinspection PyPep8Naming
    PORT_IN_ENE = 11
    # noinspection PyPep8Naming
    PORT_IN_COR = 12
    # noinspection PyPep8Naming
    PORT_IN_ENT = 13
    # noinspection PyPep8Naming
    PORT_OUTPUT = 14
    # noinspection PyPep8Naming
    PORT_IN_CON_NAME = 15
    # noinspection PyPep8Naming
    PORT_IN_DIS_NAME = 16
    # noinspection PyPep8Naming
    PORT_IN_HOM_NAME = 17
    # noinspection PyPep8Naming
    PORT_IN_ASM_NAME = 18
    # noinspection PyPep8Naming
    PORT_IN_ENE_NAME = 19
    # noinspection PyPep8Naming
    PORT_IN_COR_NAME = 20
    # noinspection PyPep8Naming
    PORT_IN_ENT_NAME = 21

    # verify that the number of parameters are OK.
    if len(param_list) != 22:
        log_error_to_console("GLCM JOB MAIN FUNCTION PARAM NOK", str(len(param_list)))
        return False
    else:
        # get needed ports
        port_in = get_port_from_wave(name=param_list[PORT_IN_POS], wave_offset=param_list[PORT_IN_WAVE])
        port_out = get_port_from_wave(name=param_list[PORT_OUTPUT])
        # log to console information
        # log_to_console('USE THIS ONLY FOR IMPORTANT MSG NOT JOB STATE, PORT STATE OR TIME')

        # check if port's you want to use are valid
        if port_in.is_valid() is True:
            try:
            # if True:
                output = skimage.feature.greycomatrix(image=port_in.arr,
                                                      distances=np.array(param_list[PORT_IN_DISTANCE]),
                                                      angles=np.array(param_list[PORT_IN_ANGLE]),
                                                      levels=param_list[PORT_IN_MATRIX_LVL],
                                                      symmetric=param_list[PORT_IN_MATRIX_SYMMETRY],
                                                      normed=param_list[PORT_IN_MATRIX_NORMED])
                port_out.arr[:] = output[:, :, -1, -1]

                # im = plt.imshow(port_out.arr, cmap=plt.get_cmap('RdPu'))
                # plt.colorbar(im)
                # plt.title('GLCM')
                # plt.show()

                if param_list[PORT_IN_CON]:
                    p_out_contrast = get_port_from_wave(name=param_list[PORT_IN_CON_NAME])
                    contrast = skimage.feature.greycoprops(output, 'contrast')
                    p_out_contrast.arr[:] = contrast[-1][-1]
                    log_to_file(p_out_contrast.arr[0].__str__())

                if param_list[PORT_IN_DIS]:
                    p_out_dissimilarity = get_port_from_wave(name=param_list[PORT_IN_DIS_NAME])
                    dissimilarity = skimage.feature.greycoprops(output, 'dissimilarity')
                    p_out_dissimilarity.arr[:] = dissimilarity[-1][-1]
                    log_to_file(p_out_dissimilarity.arr[0].__str__())

                if param_list[PORT_IN_HOM]:
                    p_out_homogeneity = get_port_from_wave(name=param_list[PORT_IN_HOM_NAME])
                    homogeneity = skimage.feature.greycoprops(output, 'homogeneity')
                    p_out_homogeneity.arr[:] = homogeneity[-1][-1]
                    log_to_file(p_out_homogeneity.arr[0].__str__())

                if param_list[PORT_IN_ASM]:
                    p_out_asm = get_port_from_wave(name=param_list[PORT_IN_ASM_NAME])
                    asm = skimage.feature.greycoprops(output, 'ASM')
                    p_out_asm.arr[:] = asm[-1][-1]
                    log_to_file(p_out_asm.arr[0].__str__())

                if param_list[PORT_IN_ENE]:
                    p_out_energy = get_port_from_wave(name=param_list[PORT_IN_ENE_NAME])
                    energy = skimage.feature.greycoprops(output, 'energy')
                    p_out_energy.arr[:] = energy[-1][-1]
                    log_to_file(p_out_energy.arr[0].__str__())

                if param_list[PORT_IN_COR]:
                    p_out_correlation = get_port_from_wave(name=param_list[PORT_IN_COR_NAME])
                    correlation = skimage.feature.greycoprops(output, 'correlation')
                    p_out_correlation.arr[:] = correlation[-1][-1]
                    log_to_file(p_out_correlation.arr[0].__str__())

                if param_list[PORT_IN_ENT]:
                    p_out_entropy = get_port_from_wave(name=param_list[PORT_IN_ENT_NAME])
                    entropy = skimage.measure.shannon_entropy(output)
                    p_out_entropy.arr[:] = entropy
                    log_to_file(p_out_entropy.arr[0].__str__())


                port_out.set_valid()
            except BaseException as error:
                log_error_to_console("GLCM JOB NOK: ", str(error))
                if param_list[PORT_IN_CON]:
                    log_to_file('')
                if param_list[PORT_IN_DIS]:
                    log_to_file('')
                if param_list[PORT_IN_HOM]:
                    log_to_file('')
                if param_list[PORT_IN_ASM]:
                    log_to_file('')
                if param_list[PORT_IN_ENE]:
                    log_to_file('')
                if param_list[PORT_IN_COR]:
                    log_to_file('')
                if param_list[PORT_IN_ENT]:
                    log_to_file('')
                pass
        else:
            return False

        return True

############################################################################################################################################
# Job create functions
############################################################################################################################################

def do_glcm_job(port_input_name: str,
                distance: list, angles: list,
                levels_matrix: int = 256, symmetric_matrix: bool = False, normed_matrix: bool = False,
                calculate_contrast: bool = False, calculate_dissimilarity: bool = False, calculate_homogeneity: bool = False,
                calculate_energy: bool = False, calculate_correlation: bool = False, calculate_ASM: bool = False, calculate_entropy: bool = False,
                port_img_output: str = None,
                level: PYRAMID_LEVEL = PYRAMID_LEVEL.LEVEL_0, wave_offset: int = 0) -> str:
    """
    Calculate the grey-level co-occurrence matrix. A grey level co-occurence matrix is a histogram of co-occuring greyscale values at a given
    offset over an image.
    http://haralick.org/journals/TexturalFeatures.pdf
    :param port_input_name: name of input port
    :param distance: list of pixel pair distance offsets.
    :param angles: list of pixel pair angles in radians.
    :param levels_matrix: indicate the number of grey-levels counted (typically 256 for an 8-bit image). The maximum value is 256.
    :param symmetric_matrix: both (i, j) and (j, i) are accumulated when (i, j) is encountered for a given offset. The default is False.
    :param normed_matrix: dividing by the total number of accumulated co-occurrences for the given offset. The elements of the resulting
                          matrix sum to 1. The default is False.
    :param calculate_contrast: if we want to calculate the contrast of the GLCM
    :param calculate_dissimilarity: if we want to calculate the dissimilarity of the GLCM
    :param calculate_homogeneity: if we want to calculate the homogeneity of the GLCM
    :param calculate_energy: if we want to calculate the energy of the GLCM
    :param calculate_correlation: if we want to calculate the correlation of the GLCM
    :param calculate_ASM: if we want to calculate the ASM of the GLCM
    :param calculate_entropy: if we want to calculate the entropy of the GLCM
    :param port_img_output: name of output port
    :param level: Level of input port, please correlate with each input port name parameter
    :param wave_offset: wave of input port, please correlate with each input port name parameter
    :return: Name of output port or ports
    """
    # Do this for each input port this function has
    input_port_name = transform_port_name_lvl(name=port_input_name, lvl=level)

    if port_img_output is None:
        port_img_output = 'GLCM_D_{dist}_A_{angle}'.format(dist=distance.__str__().replace('[', '').replace(']', ''),
                                                           angle=angles.__str__().replace('[', '').replace(']', '').replace('.', '_'),
                                                           Input=port_input_name)

        if levels_matrix != 256:
            port_img_output += '_L_' + levels_matrix.__str__()

        if symmetric_matrix:
            port_img_output += '_SYMMETRIC'

        if normed_matrix:
            port_img_output += '_NORMED'

    # size can be custom as needed
    level = PYRAMID_LEVEL.add_level(size=(levels_matrix, levels_matrix))
    port_img_output_name = transform_port_name_lvl(name=port_img_output, lvl=level)
    port_img_output_name_size = transform_port_size_lvl(lvl=level, rgb=False)

    input_port_list = [input_port_name]
    main_func_list = [input_port_name, wave_offset, distance, angles, levels_matrix, symmetric_matrix, normed_matrix,
                      calculate_contrast, calculate_dissimilarity, calculate_homogeneity, calculate_ASM, calculate_energy, calculate_correlation, calculate_entropy,
                      port_img_output_name]

    if normed_matrix:
        output_port_list = [(port_img_output_name, port_img_output_name_size, 'f', True)]
    else:
        output_port_list = [(port_img_output_name, port_img_output_name_size, 'h', True)]

    port_img_output_name = transform_port_name_lvl(name=port_img_output + '_CON', lvl=level)
    output_port_list.append((port_img_output_name, "1", 'f', False))
    main_func_list.append(port_img_output_name)

    port_img_output_name = transform_port_name_lvl(name=port_img_output + '_DIS', lvl=level)
    output_port_list.append((port_img_output_name, "1", 'f', False))
    main_func_list.append(port_img_output_name)

    port_img_output_name = transform_port_name_lvl(name=port_img_output + '_HOM', lvl=level)
    output_port_list.append((port_img_output_name, "1", 'f', False))
    main_func_list.append(port_img_output_name)

    port_img_output_name = transform_port_name_lvl(name=port_img_output + '_ASM', lvl=level)
    output_port_list.append((port_img_output_name, "1", 'f', False))
    main_func_list.append(port_img_output_name)

    port_img_output_name = transform_port_name_lvl(name=port_img_output + '_ENE', lvl=level)
    output_port_list.append((port_img_output_name, "1", 'f', False))
    main_func_list.append(port_img_output_name)

    port_img_output_name = transform_port_name_lvl(name=port_img_output + '_COR', lvl=level)
    output_port_list.append((port_img_output_name, "1", 'f', False))
    main_func_list.append(port_img_output_name)

    port_img_output_name = transform_port_name_lvl(name=port_img_output + '_ENT', lvl=level)
    output_port_list.append((port_img_output_name, "1", 'f', False))
    main_func_list.append(port_img_output_name)

    init_param_list = [calculate_contrast, calculate_dissimilarity, calculate_homogeneity, calculate_ASM,
                       calculate_energy, calculate_correlation, calculate_entropy, port_img_output_name]

    job_name = job_name_create(action='GLCM', input_list=input_port_list, wave_offset=[wave_offset], level=level)

    d = create_dictionary_element(job_module=get_module_name_from_file(__file__),
                                  job_name=job_name,
                                  input_ports=input_port_list,
                                  init_func_name='init_func_global', init_func_param=init_param_list,
                                  main_func_name='main_func',
                                  main_func_param=main_func_list,
                                  output_ports=output_port_list)

    jobs_dict.append(d)

    return port_img_output


if __name__ == "__main__":
    # If you want to run something stand-alone
    pass
