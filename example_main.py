# noinspection PyUnresolvedReferences
import Application
# noinspection PyUnresolvedReferences
import Benchmarking
# noinspection PyUnresolvedReferences
import MachineLearning
# noinspection PyUnresolvedReferences
import config_main as CONFIG
# noinspection PyUnresolvedReferences
import Utils


"""
Do not run this program. It is created only for the purpose of exampling jobs uses.
"""


def main():
    # Add images preferably in the TestData folder
    Application.set_input_image_folder('TestData/smoke_test')

    # Delete old data from the out folder
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()
    ########################################################################################################################################
    # Input jobs
    ########################################################################################################################################
    # Ports for jobs will have the following structure:
    # [INPUT NAME]_[LEVEL]
    # [OUTPUT NAME]_[LEVEL]
    # If you don't set output name parameter the default will be: [JOB NAME]_[INPUT PORT NAME]_[DETAILS OF JOB]_LEVEL
    # Job for getting one image per frame for input folder
    Application.do_get_image_job(port_output_name='ORIGINAL')
    # ###
    # JOB: Get frame
    # INPUT: None
    # OUTPUT: ORIGINAL_L0,
    # ###
    Application.do_get_image_job()
    # ###
    # JOB: Get frame
    # INPUT: None
    # OUTPUT: RAW_L0,
    # ###
    ########################################################################################################################################
    # Pyramid level processing jobs
    ########################################################################################################################################
    # Pyramid Level job to transform starting from L0 -> L4 pyramid levels
    Application.do_pyramid_level_down_job(port_input_name='ORIGINAL',
                                          port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                          port_output_name='ORIGINAL_DOWN_SAMPLE',
                                          number_of_lvl=4, is_rgb=True)
    Application.do_pyramid_level_down_job(port_input_name='RAW',
                                          port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                          number_of_lvl=4, is_rgb=True)
    # ###
    # JOB: Pyramid Level Down sample x4 RAW_L0
    # INPUT: RAW_L0,
    # OUTPUT: RAW_L1, RAW_L2, RAW_L3, RAW_L4,
    # ###
    # Pyramid Level job to transform starting from L4 -> L0 pyramid levels
    Application.do_pyramid_level_up_job(port_input_name='ORIGINAL',
                                        port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_4,
                                        port_output_name='ORIGINAL_UP_SAMPLE',
                                        number_of_lvl=4, is_rgb=True)
    Application.do_pyramid_level_up_job(port_input_name='RAW',
                                        port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_4,
                                        number_of_lvl=4, is_rgb=True)
    # ###
    # JOB: Pyramid Level Up sample 4 RAW_L4
    # INPUT: RAW_L4,
    # OUTPUT: EXPAND_RAW_L3, EXPAND_RAW_L2, EXPAND_RAW_L1, EXPAND_RAW_L0,
    # ###
    # Jobs for all levels
    levels = [CONFIG.PYRAMID_LEVEL.LEVEL_0, CONFIG.PYRAMID_LEVEL.LEVEL_1, CONFIG.PYRAMID_LEVEL.LEVEL_2,
              CONFIG.PYRAMID_LEVEL.LEVEL_3, CONFIG.PYRAMID_LEVEL.LEVEL_4]
    # levels = [CONFIG.PYRAMID_LEVEL.LEVEL_0]

    for level in levels:
        ####################################################################################################################################
        # Image processing jobs
        ####################################################################################################################################
        # Transform RGB to greyscale
        # Application.do_grayscale_transform_job(port_name_raw='RAW',
        #                                        port_name_output='GRAY_RAW',
        #                                        level=level)
        Application.do_grayscale_transform_job(port_input_name='RAW',
                                               level=level)
        # ###
        # JOB: Greyscale transform RAW L0
        # INPUT: RAW_L0,
        # OUTPUT: GRAY_RAW_L0,
        # ###
        if level != CONFIG.PYRAMID_LEVEL.LEVEL_0:
            Application.do_pyramid_level_up_job(port_input_name='GRAY_RAW',
                                                port_input_lvl=level,
                                                number_of_lvl=1)

        # Obtain max pixel value from image
        Application.do_max_pixel_image_job(port_input_name='GRAY_RAW',
                                           port_output_name='MAX_PX_RAW',
                                           level=level)
        # ###
        # JOB: Max px GRAY_RAW L0
        # INPUT: GRAY_RAW_L0,
        # OUTPUT: MAX_PX_RAW_L0,
        # ###
        Application.do_max_pixel_image_job(port_input_name='RAW',
                                           level=level)
        # ###
        # JOB: Max px RAW L0
        # INPUT: RAW_L0,
        # OUTPUT: MAX_PX_RAW_L0,
        # ###

        # Obtain median pixel value from image
        Application.do_median_pixel_image_job(port_input_name='RAW',
                                              port_output_name='MEDIAN_PX_RAW',
                                              level=level)
        # ###
        # JOB: Median pixel RAW L0
        # INPUT: RAW_L0,
        # OUTPUT: MEDIAN_PX_RAW_L0,
        # ###
        Application.do_median_pixel_image_job(port_input_name='GRAY_RAW',
                                              level=level)
        # ###
        # JOB: Median pixel GRAY_RAW L0
        # INPUT: GRAY_RAW_L0,
        # OUTPUT: MEDIAN_PX_GRAY_RAW_L0,
        # ###
        # Obtain median pixel value from image
        Application.do_mean_pixel_image_job(port_input_name='RAW',
                                            port_output_name='MEAN_PX_RAW',
                                            level=level)
        # ###
        # JOB: Mean pixel RAW L0
        # INPUT: RAW_L0,
        # OUTPUT: MEAN_PX_RAW_L0,
        # ###
        Application.do_mean_pixel_image_job(port_input_name='GRAY_RAW',
                                            level=level)
        # ###
        # JOB: Mean pixel GRAY_RAW L0
        # INPUT: GRAY_RAW_L0,
        # OUTPUT: MEAN_PX_GRAY_RAW_L0,
        # ###
        # Add  gaussian noise to an image
        Application.do_add_gaussian_blur_noise_job(port_input_name='RAW',
                                                   port_output_name='GAUSS_NOISE_RAW_MEAN_VAL_1_0_VAR_0_5',
                                                   mean_value=1,
                                                   variance=0.5,
                                                   level=level, is_rgb=True)
        # ###
        # JOB: Gaussian Noise RAW MEAN 1 VAR 0.5 L0
        # INPUT: RAW_L0,
        # OUTPUT: GAUSS_NOISE_RAW_MEAN_VAL_1_0_VAR_0_5_L0,
        # ###
        Application.do_add_gaussian_blur_noise_job(port_input_name='GRAY_RAW',
                                                   mean_value=0.5,
                                                   variance=0.5,
                                                   level=level)
        # ###
        # JOB: Gaussian Noise GRAY_RAW MEAN 0.5 VAR 0.5 L0
        # INPUT: GRAY_RAW_L0,
        # OUTPUT: GAUSS_NOISE_GRAY_RAW_MEAN_VAL_0_5_VAR_0_5_L0,
        # ###
        # Add salt and paper noise to an image
        Application.do_add_salt_pepper_noise(port_input_name='RAW',
                                             port_output_name='S&P_NOISE_RAW_DEN_0_5',
                                             density=0.5,
                                             level=level, is_rgb=True)
        # ###
        # JOB: Salt & Pepper Noise RAW DENS 0.5 L0
        # INPUT: RAW_L0,
        # OUTPUT: S&P_NOISE_RAW_DEN_0_5_L0,
        # ###
        Application.do_add_salt_pepper_noise(port_input_name='GRAY_RAW',
                                             density=0.5,
                                             level=level)
        # ###
        # JOB: Salt & Pepper Noise GRAY_RAW DENS 0.5 L0
        # INPUT: GRAY_RAW_L0,
        # OUTPUT: S & P_NOISE_GRAY_RAW_DENS_0_5_L0,
        # ###
        # Add speckle and paper noise to an image
        Application.do_add_speckle_noise(port_input_name='RAW',
                                         port_output_name='SPACKLE_NOISE_RAW_VAR_0_5',
                                         variance=0.5,
                                         level=level, is_rgb=True)
        # ###
        # JOB: Speckle Noise RAW VAR 0.5 L0
        # INPUT: RAW_L0,
        # OUTPUT: SPACKLE_NOISE_RAW_VAR_0_5_L0,
        # ###
        Application.do_add_speckle_noise(port_input_name='GRAY_RAW',
                                         variance=0.5,
                                         level=level)
        # ###
        # JOB   : Speckle Noise GRAY_RAW VAR 0.5 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: SPACKLE_NOISE_GRAY_RAW_VAR_0_5_L0,
        # ###
        # do image complement
        # Application.do_image_complement_job(port_name_input='RAW',
        #                                     port_out_name='COMPLEMENT_RAW',
        #                                     level=level)
        Application.do_image_complement_job(port_input_name='GRAY_RAW',
                                            level=level)
        # ###
        # JOB   : Complement image GRAY_RAW_L0 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: COMPLEMENT_GRAY_RAW_L0,
        # ###
        ####################################################################################################################################
        # Image blurring jobs
        ####################################################################################################################################
        # Image blurring using gaussian distribution
        Application.do_gaussian_blur_image_job(port_input_name='RAW',
                                               port_output_name='GAUSS_BLUR_RAW_K_5_S_1_4',
                                               kernel_size=5,
                                               sigma=1.4,
                                               level=level, is_rgb=True)
        # ###
        # JOB   : Gaussian Blur RAW K 5 S 1.4 L0
        # INPUT : RAW_L0,
        # OUTPUT: GAUSS_BLUR_RAW_K_5_S_1_4_L0,
        # ###
        Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW',
                                               kernel_size=9,
                                               sigma=1.4,
                                               level=level)
        # ###
        # JOB   : Gaussian Blur GRAY_RAW K 9 S 1.4 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: GAUSS_BLUR_GRAY_RAW_K_9_S_1_4_L0,
        # ###
        # Image blurring using median filter
        Application.do_median_blur_job(port_input_name='RAW',
                                       port_output_name='MEDIAN_BLUR_RAW_K_5',
                                       kernel_size=5,
                                       level=level, is_rgb=True)
        # ###
        # JOB   : Median Blur RAW K 5 L0
        # INPUT : RAW_L0,
        # OUTPUT: MEDIAN_BLUR_RAW_K_5_L0,
        # ###
        Application.do_median_blur_job(port_input_name='GRAY_RAW',
                                       kernel_size=9,
                                       level=level)
        # ###
        # JOB   : Median Blur GRAY_RAW K 9 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: MEDIAN_BLUR_GRAY_RAW_K_9_L0,
        # ###
        Application.do_mean_blur_job(port_input_name='RAW_G',
                                     kernel_size=5)
        # ###
        # JOB   : Mean Blur RAW_GRAW_G K 5 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: MEAN_BLUR_RAW_G_K_5_L0,
        # ###
        Application.do_conservative_filter_job(port_input_name='RAW_G',
                                               kernel_size=5)
        # ###
        # JOB   : Conservative filter RAW_GRAW_G K 5 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: CONSERVATIVE_RAW_G_K_5_L0,
        # ###
        Application.do_crimmins_job(port_input_name='RAW_G')
        # ###
        # JOB   : Crimmins RAW_G L0
        # INPUT : RAW_G_L0,
        # OUTPUT: CRIMMINS_RAW_G_L0,
        # ###
        Application.do_unsharp_filter_job(port_input_name='RAW_G',
                                          radius=2, percent=150)
        # ###
        # JOB   : Unsharp Filter RAW_G L0
        # INPUT : RAW_G_L0,
        # OUTPUT: UNSHARP_FILER_RAW_G_L0,
        ###
        ##################################################################################################################################
        # Image thresholding jobs
        ##################################################################################################################################
        # Do Otsu transformation on image
        # Application.do_otsu_job(input_port_name='GRAY_RAW',
        #                         output_port_name='OTSU_RAW',
        #                         level=level)
        Application.do_otsu_job(port_input_name='GRAY_RAW',
                                level=level)
        # ###
        # JOB   : Otsu Transformation GRAY_RAW L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: OTSU_GRAY_RAW_IMG_L0, OTSU_GRAY_RAW_VALUE_L0,
        # ###
        # Do image threshold on image
        Application.do_image_threshold_job(port_input_name='GRAY_RAW',
                                           input_value=150,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THR_GRAY_RAW_BINARY_150',
                                           level=level)
        # ###
        # JOB   : Image thresholding GRAY_RAW BINARY 150 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: THR_GRAY_RAW_BINARY_150_L0,
        # ###
        Application.do_image_threshold_job(port_input_name='GRAY_RAW',
                                           input_value=100,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY_INV,
                                           level=level)
        # ###
        # JOB   : Image thresholding GRAY_RAW BINARY_INV 100 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: THR_GRAY_RAW_BINARY_INV_100_L0,
        # ###
        Application.do_image_threshold_job(port_input_name='GRAY_RAW',
                                           input_value='MEAN_PX_GRAY_RAW',
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_TO_ZERO_INV,
                                           level=level)
        # ###
        # JOB   : Image thresholding GRAY_RAW TOZERO_INV MEAN_PX_GRAY_RAW_L0 L0
        # INPUT : GRAY_RAW_L0, MEAN_PX_GRAY_RAW_L0,
        # OUTPUT: THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW_L0,
        # ###
        ##################################################################################################################################
        # Multiple image jobs
        ##################################################################################################################################
        Application.do_matrix_difference_job(port_input_name_1='THR_GRAY_RAW_BINARY_150',
                                             port_input_name_2='THR_GRAY_RAW_BINARY_INV_100',
                                             port_output_name='DIFF_THR_GRAY_RAW_BINARY_150_-_THR_GRAY_RAW_BINARY_INV_100',
                                             level=level)
        # ###
        # JOB   : Matrix diff THR_GRAY_RAW_BINARY_150 - THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_150_L0, THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: DIFF_THR_GRAY_RAW_BINARY_150_-_THR_GRAY_RAW_BINARY_INV_100_L0,
        # ###
        Application.do_matrix_difference_job(port_input_name_1='THR_GRAY_RAW_BINARY_INV_100',
                                             port_input_name_2='THR_GRAY_RAW_BINARY_150',
                                             level=level)
        # ###
        # JOB   : Matrix diff THR_GRAY_RAW_BINARY_INV_100 - THR_GRAY_RAW_BINARY_150 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0,
        # OUTPUT: DIFF_THR_GRAY_RAW_BINARY_INV_100_-_THR_GRAY_RAW_BINARY_150_L0,
        # ###
        Application.do_matrix_sum_job(port_input_name_1='THR_GRAY_RAW_BINARY_150',
                                      port_input_name_2='THR_GRAY_RAW_BINARY_INV_100',
                                      port_output_name='SUM_THR_GRAY_RAW_BINARY_150_+_THR_GRAY_RAW_BINARY_INV_100',
                                      level=level)
        # ###
        # JOB   : Matrix sum THR_GRAY_RAW_BINARY_150 + THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_150_L0, THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: SUM_THR_GRAY_RAW_BINARY_150_+_THR_GRAY_RAW_BINARY_INV_100_L0,
        # ###
        Application.do_matrix_sum_job(port_input_name_1='THR_GRAY_RAW_BINARY_INV_100',
                                      port_input_name_2='THR_GRAY_RAW_BINARY_150',
                                      level=level)
        # ###
        # JOB   : Matrix sum THR_GRAY_RAW_BINARY_INV_100 + THR_GRAY_RAW_BINARY_150 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0,
        # OUTPUT: SUM_THR_GRAY_RAW_BINARY_INV_100_+_THR_GRAY_RAW_BINARY_150_L0,
        # ###
        Application.do_matrix_bitwise_and_job(port_input_name_1='THR_GRAY_RAW_BINARY_150',
                                              port_input_name_2='THR_GRAY_RAW_BINARY_INV_100',
                                              port_output_name='AND_THR_GRAY_RAW_BINARY_150_&_THR_GRAY_RAW_BINARY_INV_100',
                                              level=level)
        # ###
        # JOB   : Matrix bitwise AND of THR_GRAY_RAW_BINARY_150_L0 and THR_GRAY_RAW_BINARY_INV_100_L0 L0
        # INPUT : THR_GRAY_RAW_BINARY_150_L0, THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: AND_THR_GRAY_RAW_BINARY_150_&_THR_GRAY_RAW_BINARY_INV_100_L0,
        # ###
        Application.do_matrix_bitwise_and_job(port_input_name_1='THR_GRAY_RAW_BINARY_INV_100',
                                              port_input_name_2='THR_GRAY_RAW_BINARY_150',
                                              level=level)
        # ###
        # JOB   : Matrix bitwise AND of THR_GRAY_RAW_BINARY_INV_100_L0 and THR_GRAY_RAW_BINARY_150_L0 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0,
        # OUTPUT: AND_THR_GRAY_RAW_BINARY_INV_100_&_THR_GRAY_RAW_BINARY_150_L0,
        # ###
        Application.do_matrix_bitwise_or_job(port_input_name_1='THR_GRAY_RAW_BINARY_150',
                                             port_input_name_2='THR_GRAY_RAW_BINARY_INV_100',
                                             port_output_name='OR_THR_GRAY_RAW_BINARY_150__THR_GRAY_RAW_BINARY_INV_100',
                                             level=level)
        # ###
        # JOB   : Matrix bitwise OR of THR_GRAY_RAW_BINARY_150_L0 and THR_GRAY_RAW_BINARY_INV_100_L0 L0
        # INPUT : THR_GRAY_RAW_BINARY_150_L0, THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: OR_THR_GRAY_RAW_BINARY_150__THR_GRAY_RAW_BINARY_INV_100_L0,
        # ###
        Application.do_matrix_bitwise_or_job(port_input_name_1='THR_GRAY_RAW_BINARY_INV_100',
                                             port_input_name_2='THR_GRAY_RAW_BINARY_150',
                                             level=level)
        # ###
        # JOB   : Matrix bitwise OR of THR_GRAY_RAW_BINARY_INV_100_L0 and THR_GRAY_RAW_BINARY_150_L0 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0,
        # OUTPUT: OR_THR_GRAY_RAW_BINARY_INV_100__THR_GRAY_RAW_BINARY_150_L0,
        # ###
        Application.do_matrix_bitwise_xor_job(port_input_name_1='THR_GRAY_RAW_BINARY_150',
                                              port_input_name_2='THR_GRAY_RAW_BINARY_INV_100',
                                              port_output_name='XOR_THR_GRAY_RAW_BINARY_150_^_THR_GRAY_RAW_BINARY_INV_100',
                                              level=level)
        # ###
        # JOB   : Matrix bitwise XOR of THR_GRAY_RAW_BINARY_150_L0 and THR_GRAY_RAW_BINARY_INV_100_L0 L0
        # INPUT : THR_GRAY_RAW_BINARY_150_L0, THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: XOR_THR_GRAY_RAW_BINARY_150_^_THR_GRAY_RAW_BINARY_INV_100_L0,
        # ###
        Application.do_matrix_bitwise_xor_job(port_input_name_1='THR_GRAY_RAW_BINARY_INV_100',
                                              port_input_name_2='THR_GRAY_RAW_BINARY_150',
                                              level=level)
        # ###
        # JOB   : Matrix bitwise XOR of THR_GRAY_RAW_BINARY_INV_100_L0 and THR_GRAY_RAW_BINARY_150_L0 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0,
        # OUTPUT: XOR_THR_GRAY_RAW_BINARY_INV_100_^_THR_GRAY_RAW_BINARY_150_L0,
        # ###
        Application.do_matrix_bitwise_or_4_job(port_input_name_1='THR_GRAY_RAW_BINARY_INV_100',
                                               port_input_name_2='THR_GRAY_RAW_BINARY_150',
                                               port_input_name_3='THR_GRAY_RAW_BINARY_INV_100',
                                               port_input_name_4='THR_GRAY_RAW_BINARY_150',
                                               port_output_name='OR_THR_GRAY_RAW_BINARY_INV_100__THR_GRAY_RAW_BINARY_150__' +
                                                                'THR_GRAY_RAW_BINARY_INV_100__THR_GRAY_RAW_BINARY_150',
                                               level=level)
        # ###
        # JOB   : Matrix bitwise OR of THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0, THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0, THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0,
        # OUTPUT: OR_THR_GRAY_RAW_BINARY_INV_100__THR_GRAY_RAW_BINARY_150__THR_GRAY_RAW_BINARY_INV_100__THR_GRAY_RAW_BINARY_150_L0,
        # ###
        Application.do_matrix_bitwise_or_4_job(port_input_name_1='THR_GRAY_RAW_BINARY_INV_100',
                                               port_input_name_2='THR_GRAY_RAW_BINARY_150',
                                               port_input_name_3='THR_GRAY_RAW_BINARY_INV_100',
                                               port_input_name_4='THR_GRAY_RAW_BINARY_INV_100',
                                               level=level)
        # ###
        # JOB   : Matrix bitwise OR of THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0, THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_INV_100_L0 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_150_L0, THR_GRAY_RAW_BINARY_INV_100_L0, THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: OR_THR_GRAY_RAW_BINARY_INV_100__THR_GRAY_RAW_BINARY_150__THR_GRAY_RAW_BINARY_INV_100__THR_GRAY_RAW_BINARY_INV_100_L0,
        # ###
        ##################################################################################################################################
        # Image morphology jobs
        ##################################################################################################################################
        kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        # Erode jobs
        Application.do_image_morphological_erosion_job(port_input_name='THR_GRAY_RAW_BINARY_150',
                                                       port_output_name='MORPH_ERODED_THR_GRAY_RAW_BINARY_150_K_3_RECT_IT_1',
                                                       kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_RECTANGULAR,
                                                       kernel_size=3,
                                                       input_iteration=1,
                                                       level=level)
        # ###
        # JOB   : Image Morphological Erosion x1 THR_GRAY_RAW_BINARY_150 K 3 RECT L0
        # INPUT : THR_GRAY_RAW_BINARY_150_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_BINARY_150_K_3_RECT_IT_1_L0,
        # ###
        Application.do_image_morphological_erosion_job(port_input_name='THR_GRAY_RAW_BINARY_150',
                                                       port_output_name='MORPH_ERODED_THR_GRAY_RAW_BINARY_150_K_3_CUSTOM_1_IT_2',
                                                       kernel_to_use=kernel,
                                                       input_iteration=2,
                                                       level=level)
        # ###
        # JOB   : Image Morphological Erosion x2 THR_GRAY_RAW_BINARY_150 K 3 CUSTOM_0 L0
        # INPUT : THR_GRAY_RAW_BINARY_150_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_BINARY_150_K_3_CUSTOM_1_IT_2_L0,
        # ###
        Application.do_image_morphological_erosion_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                       kernel_to_use=kernel,
                                                       input_iteration=1,
                                                       level=level)
        # ###
        # JOB   : Image Morphological Erosion x1 THR_GRAY_RAW_BINARY_INV_100 K 3 CUSTOM_0 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_0_IT_1_L0,
        # ###
        Application.do_image_morphological_erosion_job(port_input_name='THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW',
                                                       kernel_to_use=kernel,
                                                       input_iteration=1,
                                                       level=level)
        # ###
        # JOB   : Image Morphological Erosion x1 THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW K 3 CUSTOM_0 L0
        # INPUT : THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW_K_3_CUSTOM_0_IT_1_L0,
        # ###
        Application.do_image_morphological_erosion_job(port_input_name='THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW',
                                                       kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_CROSS,
                                                       kernel_size=5,
                                                       input_iteration=2,
                                                       level=level)
        # ###
        # JOB   : Image Morphological Erosion x2 THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW K 5 CROSS L0
        # INPUT : THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW_K_5_CROSS_IT_2_L0,
        # ###
        # Dilate jobs
        Application.do_image_morphological_dilation_job(port_input_name='THR_GRAY_RAW_BINARY_150',
                                                        port_output_name='MORPH_DILATED_THR_GRAY_RAW_BINARY_150_K_3_RECT_IT_1',
                                                        kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_RECTANGULAR,
                                                        kernel_size=3,
                                                        input_iteration=1,
                                                        level=level)
        # ###
        # JOB   : Image Morphological Dilation x1 THR_GRAY_RAW_BINARY_150 K 3 RECT L0
        # INPUT : THR_GRAY_RAW_BINARY_150_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_BINARY_150_K_3_RECT_IT_1_L0,
        # ###
        Application.do_image_morphological_dilation_job(port_input_name='THR_GRAY_RAW_BINARY_150',
                                                        port_output_name='MORPH_DILATED_THR_GRAY_RAW_BINARY_150_K_3_CUSTOM_1_IT_2',
                                                        kernel_to_use=kernel,
                                                        input_iteration=2,
                                                        level=level)
        # ###
        # JOB   : Image Morphological Dilation x2 THR_GRAY_RAW_BINARY_150 K 3 CUSTOM_0 L0
        # INPUT : THR_GRAY_RAW_BINARY_150_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_BINARY_150_K_3_CUSTOM_1_IT_2_L0,
        # ###
        Application.do_image_morphological_dilation_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                        kernel_to_use=kernel,
                                                        input_iteration=1,
                                                        level=level)
        # ###
        # JOB   : Image Morphological Dilation x1 THR_GRAY_RAW_BINARY_INV_100 K 3 CUSTOM_0 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_0_IT_1_L0,
        # ###
        Application.do_image_morphological_dilation_job(port_input_name='THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW',
                                                        kernel_to_use=kernel,
                                                        input_iteration=1,
                                                        level=level)
        # ###
        # JOB   : Image Morphological Dilation x1 THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW K 3 CUSTOM_0 L0
        # INPUT : THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW_K_3_CUSTOM_0_IT_1_L0,
        # ###
        Application.do_image_morphological_dilation_job(port_input_name='THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW',
                                                        kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_CROSS,
                                                        kernel_size=5,
                                                        input_iteration=2,
                                                        level=level)
        # ###
        # JOB   : Image Morphological Dilation x2 THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW K 5 CROSS L0
        # INPUT : THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_TOZERO_INV_MEAN_PX_GRAY_RAW_K_5_CROSS_IT_2_L0,
        # ###
        # Use directly cv2 to calculate morphological operations
        Application.do_image_morphological_cv2_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                   port_output_name='MORPH_HIT_MISS_CV2_THR_GRAY_RAW_BINARY_INV_100',
                                                   level=level,
                                                   operation_to_use='cv2.MORPH_HITMISS',
                                                   input_structural_element='cv2.MORPH_RECT',
                                                   input_structural_kernel=9,
                                                   input_iteration=1,
                                                   use_custom_kernel=None)
        # ###
        # JOB   : Image Morphological job HITMISS THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_HIT_MISS_CV2_THR_GRAY_RAW_BINARY_INV_100_L0,
        # ###
        # other morphological jobs
        kernel_2 = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]
        # this will create individual jobs for each step ERODE JOB followed by an DILATION JOB.
        Application.do_image_morphological_open_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                    port_output_name='MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1',
                                                    kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_ELLIPSE,
                                                    kernel_size=9,
                                                    input_iteration=1,
                                                    level=level)
        # ###
        # JOB   : Image Morphological Erosion x1 THR_GRAY_RAW_BINARY_INV_100 K 9 ELLIPSE L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological Dilation x1 MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1 K 9 ELLIPSE L0
        # INPUT : MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0,
        # OUTPUT: MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0,
        # ###
        Application.do_image_morphological_open_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                    kernel_to_use=kernel_2,
                                                    input_iteration=1,
                                                    level=level)
        # ###
        # JOB   : Image Morphological Erosion x1 THR_GRAY_RAW_BINARY_INV_100 K 3 CUSTOM_1 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological Dilation x1 MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1 K 3 CUSTOM_1 L0
        # INPUT : MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0,
        # OUTPUT: MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0,
        # ###
        # this will create only one job
        Application.do_image_morphological_open_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                    port_output_name='MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_FAST',
                                                    kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_ELLIPSE,
                                                    kernel_size=9,
                                                    input_iteration=1,
                                                    level=level,
                                                    do_fast=True)
        # ###
        # JOB   : Image Morphological job OPEN THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_FAST_L0,
        # ###
        Application.do_image_morphological_open_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                    kernel_to_use=kernel_2,
                                                    input_iteration=1,
                                                    level=level,
                                                    do_fast=True)
        # ###
        # JOB   : Image Morphological job OPEN THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_FAST_L0,
        # ###
        # this will create individual jobs for each step DILATION JOB followed by an ERODE JOB.
        Application.do_image_morphological_close_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                     port_output_name='MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1',
                                                     kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_ELLIPSE,
                                                     kernel_size=9,
                                                     input_iteration=1,
                                                     level=level)
        # ###
        # JOB   : Image Morphological Dilation x1 THR_GRAY_RAW_BINARY_INV_100 K 9 ELLIPSE L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological Erosion x1 MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1 K 9 ELLIPSE L0
        # INPUT : MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0,
        # OUTPUT: MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0,
        # ###
        Application.do_image_morphological_close_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                     kernel_to_use=kernel_2,
                                                     input_iteration=1,
                                                     level=level)
        # ###
        # JOB   : Image Morphological Dilation x1 THR_GRAY_RAW_BINARY_INV_100 K 3 CUSTOM_1 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological Erosion x1 MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1 K 3 CUSTOM_1 L0
        # INPUT : MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0,
        # OUTPUT: MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0,
        # ###
        # this will create only one job
        Application.do_image_morphological_close_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                     port_output_name='MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_FAST  ',
                                                     kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_ELLIPSE,
                                                     kernel_size=9,
                                                     input_iteration=1,
                                                     level=level,
                                                     do_fast=True)
        # ###
        # JOB   : Image Morphological job CLOSE THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_FAST  _L0,
        # ###
        Application.do_image_morphological_close_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                     kernel_to_use=kernel_2,
                                                     input_iteration=1,
                                                     level=level,
                                                     do_fast=True)
        # ###
        # JOB   : Image Morphological job CLOSE THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_FAST_L0,
        # ###
        # this will create individual jobs for each step DILATION JOB, ERODE JOB and DIFF_DILATED_ERODE.
        Application.do_image_morphological_edge_gradient_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                             port_output_name='MORPH_EDGE_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1',
                                                             kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_ELLIPSE,
                                                             kernel_size=9,
                                                             input_iteration=1,
                                                             level=level)
        # ###
        # JOB   : Image Morphological Dilation x1 THR_GRAY_RAW_BINARY_INV_100 K 9 ELLIPSE L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological Erosion x1 THR_GRAY_RAW_BINARY_INV_100 K 9 ELLIPSE L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0,
        # ###
        # ###
        # JOB   : Matrix diff MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1 - MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1 L0
        # INPUT : MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0, MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0,
        # OUTPUT: MORPH_EDGE_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_L0,
        # ###
        Application.do_image_morphological_edge_gradient_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                             kernel_to_use=kernel_2,
                                                             input_iteration=1,
                                                             level=level)
        # ###
        # JOB   : Image Morphological Dilation x1 THR_GRAY_RAW_BINARY_INV_100 K 3 CUSTOM_1 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological Erosion x1 THR_GRAY_RAW_BINARY_INV_100 K 3 CUSTOM_1 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0,
        # ###
        # ###
        # JOB   : Matrix diff MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1 - MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1 L0
        # INPUT : MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0, MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0,
        # OUTPUT: MORPH_EDGE_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_L0,
        # ###
        # this will create only one job
        Application.do_image_morphological_edge_gradient_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                             port_output_name='MORPH_EDGE_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_FAST  ',
                                                             kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_ELLIPSE,
                                                             kernel_size=9,
                                                             input_iteration=1,
                                                             level=level,
                                                             do_fast=True)
        # ###
        # JOB   : Image Morphological job GRADIENT THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_EDGE_THR_GRAY_RAW_BINARY_INV_100_K_9_ELLIPSE_IT_1_FAST  _L0,
        # ###
        Application.do_image_morphological_edge_gradient_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                             kernel_to_use=kernel_2,
                                                             input_iteration=1,
                                                             level=level,
                                                             do_fast=True)
        # ###
        # JOB   : Image Morphological job GRADIENT THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_EDGE_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_1_IT_1_FAST_L0,
        # ###
        kernel_3 = [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1]]

        # this will create individual jobs for each step OPEN JOB and DIFF_ORIGINAL_OPEN.
        Application.do_image_morphological_top_hat_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                       port_output_name='MORPH_TOP_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1',
                                                       kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_RECTANGULAR,
                                                       kernel_size=9,
                                                       input_iteration=1,
                                                       level=level)
        # ###
        # JOB   : Image Morphological Erosion x1 THR_GRAY_RAW_BINARY_INV_100 K 9 RECT L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological Dilation x1 MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1 K 9 RECT L0
        # INPUT : MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_L0,
        # OUTPUT: MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_L0,
        # ###
        # ###
        # JOB   : Matrix diff THR_GRAY_RAW_BINARY_INV_100 - MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0, MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_L0,
        # OUTPUT: MORPH_TOP_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_L0,
        # ###
        Application.do_image_morphological_top_hat_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                       kernel_to_use=kernel_3,
                                                       input_iteration=1,
                                                       level=level)
        # ###
        # JOB   : Image Morphological Erosion x1 THR_GRAY_RAW_BINARY_INV_100 K 9 CUSTOM_2 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological Dilation x1 MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1 K 9 CUSTOM_2 L0
        # INPUT : MORPH_ERODED_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_L0,
        # OUTPUT: MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_L0,
        # ###
        # ###
        # JOB   : Matrix diff THR_GRAY_RAW_BINARY_INV_100 - MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0, MORPH_OPEN_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_L0,
        # OUTPUT: MORPH_TOP_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_L0,
        # ###
        # this will create only one job
        Application.do_image_morphological_top_hat_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                       port_output_name='MORPH_TOP_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_FAST',
                                                       kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_RECTANGULAR,
                                                       kernel_size=9,
                                                       input_iteration=1,
                                                       level=level,
                                                       do_fast=True)
        # ###
        # JOB   : Image Morphological job TOPHAT THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_TOP_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_FAST_L0,
        # ###
        Application.do_image_morphological_top_hat_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                       kernel_to_use=kernel_3,
                                                       input_iteration=1,
                                                       level=level,
                                                       do_fast=True)
        # ###
        # JOB   : Image Morphological job TOPHAT THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_TOP_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_FAST_L0,
        # ###
        # this will create individual jobs for each step CLOSE JOB and DIFF_ORIGINAL_CLOSE.
        Application.do_image_morphological_black_hat_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                         port_output_name='MORPH_BLACK_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1',
                                                         kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_RECTANGULAR,
                                                         kernel_size=9,
                                                         input_iteration=1,
                                                         level=level)
        # ###
        # JOB   : Image Morphological Dilation x1 THR_GRAY_RAW_BINARY_INV_100 K 9 RECT L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological Erosion x1 MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1 K 9 RECT L0
        # INPUT : MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_L0,
        # OUTPUT: MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_L0,
        # ###
        # ###
        # JOB   : Matrix diff MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1 - THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_L0, THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_BLACK_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_L0,
        # ###
        Application.do_image_morphological_black_hat_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                         kernel_to_use=kernel_3,
                                                         input_iteration=1,
                                                         level=level)
        # ###
        # JOB   : Image Morphological Dilation x1 THR_GRAY_RAW_BINARY_INV_100 K 9 CUSTOM_2 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological Erosion x1 MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1 K 9 CUSTOM_2 L0
        # INPUT : MORPH_DILATED_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_L0,
        # OUTPUT: MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_L0,
        # ###
        # ###
        # JOB   : Matrix diff MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1 - THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : MORPH_CLOSE_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_L0, THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_BLACK_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_L0,
        # ###
        # this will create only one job
        Application.do_image_morphological_black_hat_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                         port_output_name='MORPH_BLACK_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_FAST',
                                                         kernel_to_use=CONFIG.MORPH_CONFIG.KERNEL_RECTANGULAR,
                                                         kernel_size=9,
                                                         input_iteration=1,
                                                         level=level,
                                                         do_fast=True)
        # ###
        # JOB   : Image Morphological job BLACKHAT THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_BLACK_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_RECT_IT_1_FAST_L0,
        # ###
        Application.do_image_morphological_black_hat_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                         kernel_to_use=kernel_3,
                                                         input_iteration=1,
                                                         level=level,
                                                         do_fast=True)
        # ###
        # JOB   : Image Morphological job BLACKHAT THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_BLACK_HAT_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_IT_1_FAST_L0,
        # ###
        kernel_hit_miss = [[0, 1, 0], [-1, 1, 1], [-1, -1, 0]]
        # this will create only one job
        Application.do_morphological_hit_and_miss_transformation_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                                     port_output_name='MORPH_HIT_MISS_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM',
                                                                     use_custom_kernel=kernel_hit_miss,
                                                                     level=level)
        # ###
        # JOB   : Image Morphological job Hit Miss THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_HIT_MISS_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_L0,
        # ###
        Application.do_morphological_hit_and_miss_transformation_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                                     use_custom_kernel=CONFIG.MORPH_CONFIG.KERNEL_HIT_MISS,
                                                                     level=level)
        # ###
        # JOB   : Image Morphological job Hit Miss THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_HIT_MISS_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_3_L0,
        # ###
        Application.do_morphological_hit_and_miss_transformation_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                                     use_custom_kernel=kernel_3,
                                                                     level=level)
        # ###
        # JOB   : Image Morphological job Hit Miss THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_HIT_MISS_THR_GRAY_RAW_BINARY_INV_100_K_9_CUSTOM_2_L0,
        # ###
        kernel_thinning_1 = [[-1, -1, -1], [0, 1, 1], [1, 1, 1]]
        kernel_thinning_2 = [[0, -1, -1], [1, 1, -1], [0, 1, 0]]

        Application.do_morphological_thinning_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                  port_output_name='MORPH_THIN_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_K3_CUSTOM',
                                                  use_custom_kernel_1=CONFIG.MORPH_CONFIG.KERNEL_THINNING_1,
                                                  use_custom_kernel_2=CONFIG.MORPH_CONFIG.KERNEL_THINNING_2,
                                                  input_iteration=3,
                                                  level=level)
        # ###
        # JOB   : Image Morphological job Thinning THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_THIN_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_K3_CUSTOM_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological job Thinning MORPH_THIN_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_K3_CUSTOM_IT_1 L0
        # INPUT : MORPH_THIN_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_K3_CUSTOM_IT_1_L0,
        # OUTPUT: MORPH_THIN_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_K3_CUSTOM_IT_2_L0,
        # ###
        # ###
        # JOB   : Image Morphological job Thinning MORPH_THIN_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_K3_CUSTOM_IT_2 L0
        # INPUT : MORPH_THIN_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_K3_CUSTOM_IT_2_L0,
        # OUTPUT: MORPH_THIN_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_K3_CUSTOM_IT_3_L0,
        # ###
        Application.do_morphological_thinning_job(port_input_name='THR_GRAY_RAW_BINARY_INV_100',
                                                  use_custom_kernel_1=kernel_thinning_1,
                                                  use_custom_kernel_2=kernel_thinning_2,
                                                  input_iteration=3,
                                                  level=level)
        # ###
        # JOB   : Image Morphological job Thinning THR_GRAY_RAW_BINARY_INV_100 L0
        # INPUT : THR_GRAY_RAW_BINARY_INV_100_L0,
        # OUTPUT: MORPH_THINNING_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_6_K_3_CUSTOM_7_IT_1_L0,
        # ###
        # ###
        # JOB   : Image Morphological job Thinning MORPH_THINNING_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_6_K_3_CUSTOM_7_IT_1 L0
        # INPUT : MORPH_THINNING_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_6_K_3_CUSTOM_7_IT_1_L0,
        # OUTPUT: MORPH_THINNING_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_6_K_3_CUSTOM_7_IT_2_L0,
        # ###
        # ###
        # JOB   : Image Morphological job Thinning MORPH_THINNING_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_6_K_3_CUSTOM_7_IT_2 L0
        # INPUT : MORPH_THINNING_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_6_K_3_CUSTOM_7_IT_2_L0,
        # OUTPUT: MORPH_THINNING_THR_GRAY_RAW_BINARY_INV_100_K_3_CUSTOM_6_K_3_CUSTOM_7_IT_3_L0,
        # ###
        ####################################################################################################################################
        # Kernel processing jobs
        ####################################################################################################################################
        # see kernels from Application/Jobs/kernels.py
        # function for kernel convolution used for magnitude calculations
        Application.do_kernel_convolution_job(job_name='Convolution Kernels Sobel 3x3 GRAY_RAW',
                                              port_input_name='GRAY_RAW',
                                              input_gx='sobel_3x3_x',
                                              input_gy='sobel_3x3_y',
                                              port_output_name='SOBEL_3x3_GRAY_RAW',
                                              level=level)
        # ###
        # JOB   : Convolution Kernels Sobel 3x3 GRAY_RAW L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: Gx_SOBEL_3x3_GRAY_RAW_L0, Gy_SOBEL_3x3_GRAY_RAW_L0,
        # ###
        kernel_edge_x = [[-1, 0, 1], [-4, 0, 4], [-1, 0, 1]]
        kernel_edge_y = [[-1, -4, -1], [0, 0, 0], [1, 4, 1]]
        Application.do_kernel_convolution_job(job_name='Convolution Kernel Proposed 3x3 GRAY_RAW',
                                              port_input_name='GRAY_RAW',
                                              input_gx=kernel_edge_x,
                                              input_gy=kernel_edge_y,
                                              port_output_name='PROPOSED_3x3_GRAY_RAW',
                                              level=level)
        # ###
        # JOB   : Convolution Kernel Proposed 3x3 GRAY_RAW L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: Gx_PROPOSED_3x3_GRAY_RAW_L0, Gy_PROPOSED_3x3_GRAY_RAW_L0,
        # ###
        # function for kernel convolution used for directional edges
        Application.do_kernel_cross_convolution_job(job_name='Convolution Kernels Cross Sobel 3x3 GRAY_RAW',
                                                    port_input_name='GRAY_RAW',
                                                    kernel='sobel_3x3_x',
                                                    port_output_name='SOBEL_CROSS_3x3_GRAY_RAW',
                                                    level=level)
        # ###
        # JOB   : Convolution Kernels Cross Sobel 3x3 GRAY_RAW L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: G0_SOBEL_CROSS_3x3_GRAY_RAW_L0, G1_SOBEL_CROSS_3x3_GRAY_RAW_L0, G2_SOBEL_CROSS_3x3_GRAY_RAW_L0,
        # G3_SOBEL_CROSS_3x3_GRAY_RAW_L0, G4_SOBEL_CROSS_3x3_GRAY_RAW_L0, G5_SOBEL_CROSS_3x3_GRAY_RAW_L0,
        # G6_SOBEL_CROSS_3x3_GRAY_RAW_L0, G7_SOBEL_CROSS_3x3_GRAY_RAW_L0,
        # ###
        Application.do_kernel_cross_convolution_job(job_name='Convolution Kernels Cross Proposed GRAY_RAW',
                                                    port_input_name='GRAY_RAW',
                                                    kernel=kernel_edge_x,
                                                    port_output_name='SOBEL_PROPOSED_GRAY_RAW_K_3',
                                                    level=level)
        # ###
        # JOB   : Convolution Kernels Cross Proposed GRAY_RAW K 3 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: G0_SOBEL_PROPOSED_GRAY_RAW_K_3_L0, G1_SOBEL_PROPOSED_GRAY_RAW_K_3_L0, G2_SOBEL_PROPOSED_GRAY_RAW_K_3_L0,
        # G3_SOBEL_PROPOSED_GRAY_RAW_K_3_L0, G4_SOBEL_PROPOSED_GRAY_RAW_K_3_L0, G5_SOBEL_PROPOSED_GRAY_RAW_K_3_L0,
        # G6_SOBEL_PROPOSED_GRAY_RAW_K_3_L0, G7_SOBEL_PROPOSED_GRAY_RAW_K_3_L0,
        # ###
        Application.do_kernel_frei_chen_convolution_job(port_input_name='GRAY_RAW',
                                                        port_output_name='FREI_CHEN_3x3',
                                                        level=level)
        # ###
        # JOB   : Convolution Kernels Frei-Chen GRAY_RAW K 3 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: G0_FREI_CHEN_3x3_GRAY_RAW_L0, G1_FREI_CHEN_3x3_GRAY_RAW_L0, G2_FREI_CHEN_3x3_GRAY_RAW_L0,
        # G3_FREI_CHEN_3x3_GRAY_RAW_L0, G4_FREI_CHEN_3x3_GRAY_RAW_L0, G5_FREI_CHEN_3x3_GRAY_RAW_L0,
        # G6_FREI_CHEN_3x3_GRAY_RAW_L0, G7_FREI_CHEN_3x3_GRAY_RAW_L0, G8_FREI_CHEN_3x3_GRAY_RAW_L0,
        # ###
        Application.do_kernel_frei_chen_convolution_job(port_input_name='RAW',
                                                        level=level, is_rgb=True)
        # ###
        # JOB   : Convolution Kernels Frei-Chen GRAY_RAW K 3 L0
        # INPUT : RAW_L0,
        # OUTPUT: G0_FREI_CHEN_3x3_RAW_L0, G1_FREI_CHEN_3x3_RAW_L0, G2_FREI_CHEN_3x3_RAW_L0, G3_FREI_CHEN_3x3_RAW_L0,
        # G4_FREI_CHEN_3x3_RAW_L0, G5_FREI_CHEN_3x3_RAW_L0, G6_FREI_CHEN_3x3_RAW_L0, G7_FREI_CHEN_3x3_RAW_L0,
        # G8_FREI_CHEN_3x3_RAW_L0,
        # ###
        Application.do_kernel_navatia_babu_convolution_job(port_input_name='GRAY_RAW',
                                                           port_output_name='NAVATI_BABU_5x5',
                                                           level=level)
        # ###
        # JOB   : Convolution Kernels Navatia-Babu GRAY_RAW K 5 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: G0_NAVATIA_BABU_5x5_GRAY_RAW_L0, G1_NAVATIA_BABU_5x5_GRAY_RAW_L0, G2_NAVATIA_BABU_5x5_GRAY_RAW_L0,
        # G3_NAVATIA_BABU_5x5_GRAY_RAW_L0, G4_NAVATIA_BABU_5x5_GRAY_RAW_L0, G5_NAVATIA_BABU_5x5_GRAY_RAW_L0,
        # ###
        Application.do_kernel_navatia_babu_convolution_job(port_input_name='RAW',
                                                           level=level, is_rgb=True)
        # ###
        # JOB   : Convolution Kernels Navatia-Babu GRAY_RAW K 5 L0
        # INPUT : RAW_L0,
        # OUTPUT: G0_NAVATIA_BABU_5x5_RAW_L0, G1_NAVATIA_BABU_5x5_RAW_L0, G2_NAVATIA_BABU_5x5_RAW_L0,
        # G3_NAVATIA_BABU_5x5_RAW_L0, G4_NAVATIA_BABU_5x5_RAW_L0, G5_NAVATIA_BABU_5x5_RAW_L0,
        # ###
        ####################################################################################################################################
        # edge detection - magnitude gradient jobs
        ####################################################################################################################################
        # make sure the kernels job is done if you want do use this
        Application.do_gradient_magnitude_job(job_name='PROPOSED 3x3 GRAY_RAW',
                                              port_input_name_gx='Gx_PROPOSED_3x3_GRAY_RAW',
                                              port_input_name_gy='Gy_PROPOSED_3x3_GRAY_RAW',
                                              port_output_name='PROPOSED_3x3_GRAY_RAW',
                                              level=level)
        # ###
        # JOB   : PROPOSED 3x3 GRAY_RAW L0
        # INPUT : Gx_PROPOSED_GRAY_RAW_K_3_L0, Gy_PROPOSED_GRAY_RAW_K_3_L0,
        # OUTPUT: PROPOSED_3x3_GRAY_RAW_L0,
        # ###
        Application.do_first_order_derivative_operators(port_input_name='GRAY_RAW',
                                                        operator=CONFIG.FILTERS.PREWITT_3x3,
                                                        level=level)
        # ###
        # JOB   : Convolution Kernel PREWITT 3x3 GRAY_RAW L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: Gx_PREWITT_3x3_L0, Gy_PREWITT_3x3_L0,
        # ###
        # all first derivative filters
        first_order_derivative_filters = [CONFIG.FILTERS.ROBERTS_2x2, CONFIG.FILTERS.PIXEL_DIFF_3x3,
                                          CONFIG.FILTERS.PIXEL_DIFF_SEPARATED_3x3,
                                          CONFIG.FILTERS.SOBEL_3x3, CONFIG.FILTERS.SOBEL_5x5, CONFIG.FILTERS.SOBEL_DILATED_5x5,
                                          CONFIG.FILTERS.SOBEL_7x7, CONFIG.FILTERS.SOBEL_DILATED_7x7, CONFIG.FILTERS.PREWITT_3x3,
                                          CONFIG.FILTERS.PREWITT_5x5, CONFIG.FILTERS.PREWITT_DILATED_5x5,
                                          CONFIG.FILTERS.PREWITT_LEVKINE_5x5,
                                          CONFIG.FILTERS.PREWITT_7x7, CONFIG.FILTERS.PREWITT_DILATED_7x7, CONFIG.FILTERS.SCHARR_3x3,
                                          CONFIG.FILTERS.FREI_CHEN_3x3, CONFIG.FILTERS.SCHARR_5x5, CONFIG.FILTERS.SCHARR_DILATED_5x5,
                                          CONFIG.FILTERS.KIRSCH_3x3, CONFIG.FILTERS.KIRSCH_5x5, CONFIG.FILTERS.KAYYALI_3x3,
                                          CONFIG.FILTERS.EL_ARWADI_EL_ZAART_5x5]

        for edge in first_order_derivative_filters:
            Application.do_first_order_derivative_operators(port_input_name='GRAY_RAW',
                                                            operator=edge,
                                                            level=level)
        ####################################################################################################################################
        # edge detection - directional gradient jobs
        ####################################################################################################################################
        Application.do_gradient_magnitude_cross_job(job_name='Sobel Compass 3x3 GRAY_RAW',
                                                    port_input_name='SOBEL_CROSS_3x3_GRAY_RAW',
                                                    port_output_name='SOBEL_COMPASS_3x3_GRAY_RAW',
                                                    level=level)
        # ###
        # JOB   : Sobel Compass 3x3 GRAY_RAW SOBEL_CROSS_3x3_GRAY_RAW L0
        # INPUT : G0_SOBEL_CROSS_3x3_GRAY_RAW_L0, G1_SOBEL_CROSS_3x3_GRAY_RAW_L0, G2_SOBEL_CROSS_3x3_GRAY_RAW_L0,
        # G3_SOBEL_CROSS_3x3_GRAY_RAW_L0, G4_SOBEL_CROSS_3x3_GRAY_RAW_L0, G5_SOBEL_CROSS_3x3_GRAY_RAW_L0,
        # G6_SOBEL_CROSS_3x3_GRAY_RAW_L0, G7_SOBEL_CROSS_3x3_GRAY_RAW_L0,
        # OUTPUT: SOBEL_COMPASS_3x3_GRAY_RAW_L0,
        # ###
        Application.do_gradient_navatia_babu_job(port_input_name='NAVATIA_BABU_5x5_GRAY_RAW',
                                                 port_output_name='NAVATIA_BABU_5x5_GRAY_RAW',
                                                 level=level)
        # ###
        # JOB   : Navatia-Babu 5x5 GRAY_RAW NAVATI_BABU_5x5_GRAY_RAW L0
        # INPUT : G0_NAVATIA_BABU_5x5_GRAY_RAW_L0, G1_NAVATIA_BABU_5x5_GRAY_RAW_L0, G2_NAVATIA_BABU_5x5_GRAY_RAW_L0,
        #         G3_NAVATIA_BABU_5x5_GRAY_RAW_L0, G4_NAVATIA_BABU_5x5_GRAY_RAW_L0, G5_NAVATIA_BABU_5x5_GRAY_RAW_L0,
        # OUTPUT: NAVATI_BABU_5x5_GRAY_RAW_L0,
        # ###
        Application.do_gradient_frei_chen_job(port_input_name='FREI_CHEN_3x3_GRAY_RAW',
                                              port_output_edge_name='FREI_CHEN_3x3_GRAY_RAW',
                                              port_output_line_name='FREI_CHEN_3x3_GRAY_RAW',
                                              level=level)

        compass_filters = [CONFIG.FILTERS.ROBINSON_CROSS_3x3, CONFIG.FILTERS.ROBINSON_MODIFIED_CROSS_3x3,
                           CONFIG.FILTERS.KIRSCH_CROSS_3x3, CONFIG.FILTERS.PREWITT_CROSS_3x3]
        for edge in compass_filters:
            Application.do_compass_edge_job(port_input_name='GRAY_RAW',
                                            operator=edge,
                                            level=level)
        # ###
        # JOB   : ROBINSON MODIFIED CROSS 3x3 GRAY_RAW ROBINSON_MODIFIED_CROSS_3x3 L0
        # INPUT : G0_ROBINSON_MODIFIED_CROSS_3x3_L0, G1_ROBINSON_MODIFIED_CROSS_3x3_L0, G2_ROBINSON_MODIFIED_CROSS_3x3_L0,
        # G3_ROBINSON_MODIFIED_CROSS_3x3_L0, G4_ROBINSON_MODIFIED_CROSS_3x3_L0, G5_ROBINSON_MODIFIED_CROSS_3x3_L0,
        # G6_ROBINSON_MODIFIED_CROSS_3x3_L0, G7_ROBINSON_MODIFIED_CROSS_3x3_L0,
        # OUTPUT: ROBINSON MODIFIED CROSS 3x3_GRAY_RAW_L0,
        # ###
        Application.do_canny_from_kernel_convolution_job(kernel_convolution='SOBEL_3x3_GRAY_RAW',
                                                         config_canny_threshold=CONFIG.CANNY_VARIANTS.FIX_THRESHOLD,
                                                         config_canny_threshold_value=None,
                                                         port_output_name='CANNY_FIX_THR_SOBEL_3x3_GRAY_RAW',
                                                         level=level)
        # ###
        # JOB   : Canny Fix Thr Sobel 3x3 Gray Raw using SOBEL_3x3_GRAY_RAW_L0 L0
        # INPUT : Gx_SOBEL_3x3_GRAY_RAW_L0, Gy_SOBEL_3x3_GRAY_RAW_L0,
        # OUTPUT: CANNY_FIX_THR_SOBEL_3x3_GRAY_RAW_L0,
        # ###
        Application.do_canny_config_job(port_input_name='GRAY_RAW',
                                        edge_detector=CONFIG.FILTERS.KIRSCH_3x3,
                                        canny_config=CONFIG.CANNY_VARIANTS.MEDIAN_SIGMA,
                                        canny_config_value='MEDIAN_PX_GRAY_RAW',
                                        port_output_name='CANNY_MEDIAN_SIGMA_KIRSCH_3x3_MEDIAN_PX_GRAY_RAW',
                                        level=level,
                                        do_blur=True, kernel_blur_size=5, sigma=2.0,
                                        do_otsu=False)
        # ###
        # JOB   : Gaussian Blur GRAY_RAW K 5 S 2.0 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: GRAY_RAW_GAUS_BLUR_K_5_S_2_0_L0,
        # ###
        # ###
        # JOB   : Kernel convolution KIRSCH_3x3 L0
        # INPUT : GRAY_RAW_GAUS_BLUR_K_5_S_2_0_L0,
        # OUTPUT: Gx_GRAY_RAW_GAUS_BLUR_K_5_S_2_0_KIRSCH_3x3_L0, Gy_GRAY_RAW_GAUS_BLUR_K_5_S_2_0_KIRSCH_3x3_L0,
        # ###
        # ###
        # JOB   : Canny Median Sigma Kirsch 3x3 Median Px Gray Raw using GRAY_RAW_GAUS_BLUR_K_5_S_2_0_KIRSCH_3x3_L0 L0
        # INPUT : Gx_GRAY_RAW_GAUS_BLUR_K_5_S_2_0_KIRSCH_3x3_L0, Gy_GRAY_RAW_GAUS_BLUR_K_5_S_2_0_KIRSCH_3x3_L0,
        # OUTPUT: CANNY_MEDIAN_SIGMA_KIRSCH_3x3_MEDIAN_PX_GRAY_RAW_L0,
        # ###
        # output format: CANNY_[VARIANT]_[EDGE FILTER]_[VALUE]_[INPUT]
        Application.do_canny_config_job(port_input_name='GRAY_RAW',
                                        edge_detector=CONFIG.FILTERS.KIRSCH_3x3,
                                        canny_config=CONFIG.CANNY_VARIANTS.OTSU_MEDIAN_SIGMA,
                                        canny_config_value=None,
                                        level=level,
                                        do_blur=True, kernel_blur_size=7, sigma=2.0,
                                        do_otsu=True)
        # ###
        # JOB   : Gaussian Blur GRAY_RAW K 7 S 2.0 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: GRAY_RAW_GAUS_BLUR_K_7_S_2_0_L0,
        # ###
        # ###
        # JOB   : Otsu Transformation GRAY_RAW_GAUS_BLUR_K_7_S_2_0 L0
        # INPUT : GRAY_RAW_GAUS_BLUR_K_7_S_2_0_L0,
        # OUTPUT: OTSU_GRAY_RAW_GAUS_BLUR_K_7_S_2_0_IMG_L0, OTSU_GRAY_RAW_GAUS_BLUR_K_7_S_2_0_VALUE_L0,
        # ###
        # ###
        # JOB   : Kernel convolution KIRSCH_3x3 L0
        # INPUT : GRAY_RAW_GAUS_BLUR_K_7_S_2_0_L0,
        # OUTPUT: Gx_GRAY_RAW_GAUS_BLUR_K_7_S_2_0_KIRSCH_3x3_L0, Gy_GRAY_RAW_GAUS_BLUR_K_7_S_2_0_KIRSCH_3x3_L0,
        # ###
        # ###
        # JOB   : Canny Otsu Median Sigma Kirsch 3x3 Otsu Gray Raw Gaus Blur K 7 S 2 0 Value Gray Raw Gaus Blur K 7 S 2 0 using GRAY_RAW_GAUS_BLUR_K_7_S_2_0_KIRSCH_3x3_L0 L0
        # INPUT : Gx_GRAY_RAW_GAUS_BLUR_K_7_S_2_0_KIRSCH_3x3_L0, Gy_GRAY_RAW_GAUS_BLUR_K_7_S_2_0_KIRSCH_3x3_L0,
        # OUTPUT: CANNY_OTSU_MEDIAN_SIGMA_KIRSCH_3x3_OTSU_GRAY_RAW_GAUS_BLUR_K_7_S_2_0_VALUE GRAY_RAW_GAUS_BLUR_K_7_S_2_0_L0,
        # ###
        Application.do_canny_fix_threshold_job(port_input_name='GRAY_RAW', low_manual_threshold=220, high_manual_threshold=240)
        # ###
        # JOB   : Gaussian Blur GRAY_RAW K 3 S 1 L0
        # INPUT : GRAY_RAW_L0,
        # OUTPUT: GAUS_BLUR_K_3_S_1_GRAY_RAW_L0,
        # ###
        # ###
        # JOB   : Convolution Kernel SOBEL 3x3 GAUS_BLUR_K_3_S_1_GRAY_RAW L0
        # INPUT : GAUS_BLUR_K_3_S_1_GRAY_RAW_L0,
        # OUTPUT: Gx_SOBEL_3x3_GAUS_BLUR_K_3_S_1_GRAY_RAW_L0, Gy_SOBEL_3x3_GAUS_BLUR_K_3_S_1_GRAY_RAW_L0,
        # ###
        # ###
        # JOB   : Canny Fix Threshold Sobel 3x3 220 240 Gaus Blur K 3 S 1 Gray Raw L0
        # INPUT : Gx_SOBEL_3x3_GAUS_BLUR_K_3_S_1_GRAY_RAW_L0, Gy_SOBEL_3x3_GAUS_BLUR_K_3_S_1_GRAY_RAW_L0,
        # OUTPUT: CANNY_FIX_THRESHOLD_SOBEL_3x3_220_240_GAUS_BLUR_K_3_S_1_GRAY_RAW_L0,
        # ###
        ####################################################################################################################################
        # edge detection - second derivative
        ####################################################################################################################################
        # Application.do_laplacian_pyramid_from_img_diff_job(port_input_name_1='GRAY_RAW',
        #                                                    port_input_name_2='EXPAND_GRAY_RAW',
        #                                                    port_out_name='LAPLACE_PYRAMID_GRAY_RAW',
        #                                                    level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        # ###
        # JOB   : Laplace Pyramid Img Diff GRAY_RAW_L0 and EXPAND_GRAY_RAW_L0 L0
        # INPUT : GRAY_RAW_L0, EXPAND_GRAY_RAW_L0,
        # OUTPUT: LAPLACE_PYRAMID_GRAY_RAW_L0,
        # ###
        Application.do_laplacian_pyramid_from_img_diff_job(port_input_name_1='RAW',
                                                           port_input_name_2='EXPAND_RAW',
                                                           level=level, is_rgb=True)
        # ###
        # JOB   : Laplace Pyramid Img Diff RAW_L4 and EXPAND_RAW_L4 L4
        # INPUT : RAW_L4, EXPAND_RAW_L4,
        # OUTPUT: LAPLACE_PYRAMID_ RAW_L4,
        # ###
        Application.do_laplace_job(port_input_name='GRAY_RAW',
                                   kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1,
                                   port_output_name='LAPLACE_EDGE_V1')
        # ###
        # JOB   : Laplace Edge of  RAW_G L0
        # INPUT : RAW_G_L0,
        # OUTPUT: LAPLACE_EDGE_V1_L0,
        # ###
        Application.do_log_job(port_input_name='GRAY_RAW',
                               gaussian_kernel_size=7, gaussian_sigma=8,
                               laplacian_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2)
        # ###
        # JOB   : Gaussian Blur RAW_G K 7 S 8 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: GAUS_BLUR_K_7_S_8_RAW_G_L0,
        # ###
        # ###
        # JOB   : Laplace Edge of  GAUS_BLUR_K_7_S_8_RAW_G L0
        # INPUT : GAUS_BLUR_K_7_S_8_RAW_G_L0,
        # OUTPUT: LOG_LAPLACE_3x3_V2_GAUS_BLUR_K_7_S_8_RAW_G_L0,
        # ###
        Application.do_log_job(port_input_name='GRAY_RAW',
                               laplacian_kernel=CONFIG.FILTERS_SECOND_ORDER.LOG_5x5_V1,
                               use_precalculated_kernel=True)
        # ###
        # JOB   : Laplace Edge of  RAW_G L0
        # INPUT : RAW_G_L0,
        # OUTPUT: LOG_5x5_V1_L0,
        # ###
        Application.do_zero_crossing_job(port_input_name='LOG_LAPLACE_3x3_V2_GAUS_BLUR_K_7_S_8_RAW_G')
        # ###
        # JOB   : Zero Crossing  LOG_LAPLACE_3x3_V2_GAUS_BLUR_K_7_S_8_RAW_G_L0 L0
        # INPUT : LOG_LAPLACE_3x3_V2_GAUS_BLUR_K_7_S_8_RAW_G_L0,
        # OUTPUT: ZERO_CROSSING_LOG_LAPLACE_3x3_V2_GAUS_BLUR_K_7_S_8_RAW_G_L0,
        # ###
        Application.do_marr_hildreth_job(port_input_name='GRAY_RAW',
                                         gaussian_kernel_size=7, gaussian_sigma=8,
                                         laplacian_kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1)
        # ###
        # JOB   : Marr Hildreth   RAW_G L0
        # INPUT : RAW_G_L0,
        # OUTPUT: Marr_Hildreth_RAW_G_L0,
        # ###
        Application.do_marr_hildreth_job(port_input_name='GRAY_RAW',
                                         laplacian_kernel=CONFIG.FILTERS_SECOND_ORDER.LOG_5x5_V1,
                                         use_precalculated_kernel=True)
        # ###
        # JOB   : Marr Hildreth   RAW_G L0
        # INPUT : RAW_G_L0,
        # OUTPUT: Marr_Hildreth_RAW_G_L0,
        # ###
        Application.do_dog_job(port_input_name='GRAY_RAW')
        # ###
        # JOB   : Gaussian Blur RAW_G K 3 S 1.0 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: GAUS_BLUR_K_3_S_1_0_RAW_G_L0,
        # ###
        # ###
        # JOB   : Gaussian Blur RAW_G K 5 S 1.0 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: GAUS_BLUR_K_5_S_1_0_RAW_G_L0,
        # ###
        # ###
        # JOB   : Matrix diff GAUS_BLUR_K_3_S_1_0_RAW_G - GAUS_BLUR_K_5_S_1_0_RAW_G L0
        # INPUT : GAUS_BLUR_K_3_S_1_0_RAW_G_L0, GAUS_BLUR_K_5_S_1_0_RAW_G_L0,
        # OUTPUT: DoG_GAUS_BLUR_K_3_S_1_0_GAUS_BLUR_K_5_S_1_0_L0,
        # ###
        Application.do_dog_job(port_input_name='GRAY_RAW',
                               gaussian_kernel_size_1=5, gaussian_sigma_1=1.4,
                               gaussian_kernel_size_2=7, gaussian_sigma_2=2,
                               port_output_name='DoG_RAW_G')
        # ###
        # JOB   : Gaussian Blur RAW_G K 5 S 1.4 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: GAUS_BLUR_K_5_S_1_4_RAW_G_L0,
        # ###
        # ###
        # JOB   : Gaussian Blur RAW_G K 7 S 2 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: GAUS_BLUR_K_7_S_2_RAW_G_L0,
        # ###
        # ###
        # JOB   : Matrix diff GAUS_BLUR_K_5_S_1_4_RAW_G - GAUS_BLUR_K_7_S_2_RAW_G L0
        # INPUT : GAUS_BLUR_K_5_S_1_4_RAW_G_L0, GAUS_BLUR_K_7_S_2_RAW_G_L0,
        # OUTPUT: DoG_RAW_G_L0,
        # ###
        Application.do_dob_job(port_input_name='GRAY_RAW')
        # ###
        # JOB   : Mean Blur RAW_GRAW_G K 5 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: MEAN_K_5_RAW_G_L0,
        # ###
        # ###
        # JOB   : Mean Blur RAW_GRAW_G K 3 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: MEAN_K_3_RAW_G_L0,
        # ###
        # ###
        # JOB   : Matrix diff MEAN_K_5_RAW_G - MEAN_K_3_RAW_G L0
        # INPUT : MEAN_K_5_RAW_G_L0, MEAN_K_3_RAW_G_L0,
        # OUTPUT: DoB_MEAN_K_5_RAW_MEAN_K_3_RAW_G_L0,
        # ###
        Application.do_dob_job(port_input_name='GRAY_RAW',
                               kernel_size_1=7, kernel_size_2=5,
                               port_output_name='DoB_RAW_G')
        # ###
        # JOB   : Mean Blur RAW_GRAW_G K 7 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: MEAN_K_7_RAW_G_L0,
        # ###
        # ###
        # JOB   : Mean Blur RAW_GRAW_G K 5 L0
        # INPUT : RAW_G_L0,
        # OUTPUT: MEAN_K_5_RAW_G_L0,
        # ###
        # ###
        # JOB   : Matrix diff MEAN_K_7_RAW_G - MEAN_K_5_RAW_G L0
        # INPUT : MEAN_K_7_RAW_G_L0, MEAN_K_5_RAW_G_L0,
        # OUTPUT: DoB_RAW_G_L0,
        # ###
        ####################################################################################################################################
        # line/shape detection
        ####################################################################################################################################
        Application.do_hough_lines_job(port_input_name='CANNY', vote_threshold=80,
                                       distance_resolution=1,
                                       angle_resolution=3.14159 / 180,
                                       min_line_length=20, max_line_gap=10,
                                       level=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                       overlay=True)
        # ###
        # JOB   : Hough Lines Transform of CANNY on L1 W-0
        # INPUT : CANNY_L1,
        # OUTPUT: HOUGH_LINE_CANNY_IMG_L1, HOUGH_LINE_CANNY_ARRAY_L1,
        # ###

        Application.do_hough_lines_job(port_input_name='CANNY', vote_threshold=80,
                                       distance_resolution=1,
                                       level=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                       overlay=False)
        # ###
        # JOB   : Hough Lines Transform of CANNY on L1 W-0
        # INPUT : CANNY_L1,
        # OUTPUT: HOUGH_LINE_CANNY_IMG_L1, HOUGH_LINE_CANNY_ARRAY_L1,
        # ###
        ####################################################################################################################################
        # Line/edge connectivity jobs
        ####################################################################################################################################
        Application.do_edge_label_job(port_input_name='CANNY_FIX_THR_SOBEL_3x3_GRAY_RAW',
                                      port_output_name='EDGE_LABELED_CANNY_FIX_THR_SOBEL_3x3_GRAY_RAW',
                                      level=level)
    # create config file for application
    Application.create_config_file()

    # What ports to save
    # No arguments will save all pictures
    # Application.configure_save_pictures(location='DEFAULT', ports_to_save=['RAW_L0','RAW_GRAY_L0'])
    # list_to_save = Application.create_list_ports_with_word('CANNY')
    # Application.configure_save_pictures(location='DEFAULT', ports_to_save=list_to_save)
    Application.configure_save_pictures(location='DEFAULT', ports_to_save='ALL')

    # What ports to show
    # No arguments will show all pictures for ever
    # list_to_show = Application.create_list_ports_with_word('CANNY')
    # Application.configure_save_pictures(time_to_show=150, ports_to_show=list_to_show)
    # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=100)

    # run application
    Application.run_application()

    # Do bsds benchmarking
    # Please run this only on ubuntu/linux
    # Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/out',
    #                                            gt_location='TestData/BSR/BSDS500/data/groundTruth/buildings',
    #                                            raw_image='TestData/BSR/BSDS500/data/images/buildings',
    #                                            jobs_set=['CANNY_RATIO_TRH_SOBEL_3x3_L0'])


if __name__ == "__main__":
    main()
