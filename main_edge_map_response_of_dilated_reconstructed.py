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
Code for paper:
  title={Edge map response of dilated and reconstructed classical filters},
  author={Orhei, Ciprian and Bogdan, Victor and Bonchi{\c{s}}, Cosmin},
  booktitle={2020 22nd International Symposium on Symbolic and Numeric Algorithms for Scientific Computing (SYNASC)},
  pages={187--194},
  year={2020},
  organization={IEEE}
"""


def main():
    """
    Main function of framework Please look in example_main for all functions you can use
    """
    Application.set_input_image_folder('TestData/dilation_test/test')

    list_to_save = []
    list_to_benchmark = []

    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    list_to_save.append('RAW_L0')
    Application.do_grayscale_transform_job(port_input_name='RAW')
    list_to_save.append('GRAY_RAW_L0')
    kernel = 3
    sigma = 1

    edges = [
        CONFIG.FILTERS.SOBEL_3x3,
        CONFIG.FILTERS.PREWITT_3x3,
        CONFIG.FILTERS.KIRSCH_3x3,
        CONFIG.FILTERS.SCHARR_3x3,
        CONFIG.FILTERS.KAYYALI_3x3,
        CONFIG.FILTERS.KROON_3x3
    ]

    dilated_edges_5 = [
        CONFIG.FILTERS.SOBEL_DILATED_5x5,
        CONFIG.FILTERS.PREWITT_DILATED_5x5,
        CONFIG.FILTERS.KIRSCH_DILATED_5x5,
        CONFIG.FILTERS.SCHARR_DILATED_5x5,
        CONFIG.FILTERS.KAYYALI_DILATED_5x5,
        CONFIG.FILTERS.KROON_DILATED_5x5
    ]

    dilated_edges_7 = [
        CONFIG.FILTERS.SOBEL_DILATED_7x7,
        CONFIG.FILTERS.PREWITT_DILATED_7x7,
        CONFIG.FILTERS.KIRSCH_DILATED_7x7,
        CONFIG.FILTERS.SCHARR_DILATED_7x7, CONFIG.FILTERS.KAYYALI_DILATED_7x7,
        CONFIG.FILTERS.KROON_DILATED_7x7
    ]

    ########################################################################################################################################
    ########################################################################################################################################
    # FOR LVL1 -> LVL0
    ########################################################################################################################################
    ########################################################################################################################################
    Application.do_pyramid_level_down_job(port_input_name='GRAY_RAW', number_of_lvl=1,
                                          port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0)

    Application.do_pyramid_level_up_job(port_input_name='GRAY_RAW',
                                        port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                        number_of_lvl=1)

    for index in range(len(edges)):
        ####################################################################################################################################
        # edge gradient
        ####################################################################################################################################
        # Apply Gradient 3x3 on L0
        Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', port_output_name='GAUSS_GREY_RAW',
                                               kernel_size=kernel, sigma=sigma,
                                               level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_first_order_derivative_operators(port_input_name='GAUSS_GREY_RAW',
                                                        operator=edges[index],
                                                        port_output_name=edges[index])

        Application.do_image_threshold_job(port_input_name=edges[index],
                                           input_value=60,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THR_' + edges[index],
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_thinning_guo_hall_image_job(port_input_name='THR_' + edges[index],
                                                   port_output_name='THINNED_THR_' + edges[index])

        list_to_save.append('THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)

        # Apply Gradient 5x5 on L0
        Application.do_first_order_derivative_operators(port_input_name='GAUSS_GREY_RAW',
                                                        operator=dilated_edges_5[index],
                                                        port_output_name=dilated_edges_5[index])

        Application.do_image_threshold_job(port_input_name=dilated_edges_5[index],
                                           input_value=60,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THR_' + dilated_edges_5[index],
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_thinning_guo_hall_image_job(port_input_name='THR_' + dilated_edges_5[index],
                                                   port_output_name='THINNED_THR_' + dilated_edges_5[index])

        list_to_save.append('THINNED_THR_' + dilated_edges_5[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('THINNED_THR_' + dilated_edges_5[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)

        # Apply Gradient 3x3 on L1
        Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', port_output_name='GAUSS_GREY_RAW',
                                               kernel_size=kernel, sigma=sigma,
                                               level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_first_order_derivative_operators(port_input_name='GAUSS_GREY_RAW',
                                                        operator=edges[index],
                                                        port_output_name=edges[index],
                                                        level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_image_threshold_job(port_input_name=edges[index],
                                           input_value=60,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THR_' + edges[index],
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_thinning_guo_hall_image_job(port_input_name='THR_' + edges[index],
                                                   port_output_name='THINNED_THR_' + edges[index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        list_to_save.append('THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)
        ####################################################################################################################################
        # Canny
        ####################################################################################################################################
        # Apply canny normal on L0
        Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', port_output_name='GAUSS_GREY_RAW',
                                               kernel_size=kernel, sigma=sigma,
                                               level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_max_pixel_image_job(port_input_name='GAUSS_GREY_RAW', level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_canny_ratio_threshold_job(port_input_name='GRAY_RAW',
                                                 edge_detector=edges[index],
                                                 canny_config_value='MAX_PX_GAUSS_GREY_RAW',
                                                 port_output_name='CANNY_' + edges[index],
                                                 do_blur=True, kernel_blur_size=kernel, sigma=sigma)

        list_to_save.append('CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)

        # Apply canny with dilated sobel 5x5 on L0
        Application.do_canny_ratio_threshold_job(port_input_name='GRAY_RAW',
                                                 edge_detector=dilated_edges_5[index],
                                                 canny_config_value='MAX_PX_GAUSS_GREY_RAW',
                                                 port_output_name='CANNY_' + dilated_edges_5[index],
                                                 do_blur=True, kernel_blur_size=kernel, sigma=sigma)

        list_to_save.append('CANNY_' + dilated_edges_5[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('CANNY_' + dilated_edges_5[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)

        # Apply canny with sobel 3x3 on L1
        Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', port_output_name='GAUSS_GREY_RAW',
                                               kernel_size=kernel, sigma=sigma,
                                               level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_max_pixel_image_job(port_input_name='GAUSS_GREY_RAW', level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_canny_ratio_threshold_job(port_input_name='GRAY_RAW',
                                                 edge_detector=edges[index],
                                                 canny_config_value='MAX_PX_GAUSS_GREY_RAW',
                                                 level=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                                 port_output_name='CANNY_' + edges[index],
                                                 do_blur=True, kernel_blur_size=kernel, sigma=sigma)

        list_to_save.append('CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)

        # ##################################################################################################################################
        # # go with gradient dilated results down one level to compare. We want to see the differences
        # ##################################################################################################################################
        Application.do_pyramid_level_down_job(port_input_name='THINNED_THR_' + dilated_edges_5[index], number_of_lvl=1,
                                              port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                              port_output_name='THINNED_THR_' + dilated_edges_5[index] + '_FROM_L0')

        list_to_save.append('THINNED_THR_' + dilated_edges_5[index] + '_FROM_L0' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)

        # a line will result in min 255/4~60 pixel value in lowe levels
        Application.do_image_threshold_job(port_input_name='THINNED_THR_' + dilated_edges_5[index] + '_FROM_L0',
                                           input_value=60,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THR_THINNED_THR_' + dilated_edges_5[index] + '_FROM_L0',
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        list_to_save.append('THR_THINNED_THR_' + dilated_edges_5[index] + '_FROM_L0' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_thinning_guo_hall_image_job(port_input_name='THR_THINNED_THR_' + dilated_edges_5[index] + '_FROM_L0',
                                                   port_output_name='THINNED_THR_THINNED_THR_' + dilated_edges_5[index] + '_FROM_L0',
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        list_to_save.append('THINNED_THR_THINNED_THR_' + dilated_edges_5[index] + '_FROM_L0' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)

        # Let's see the differences of images taking in consideration 1px offset in each direction because of the level manipulation
        Application.do_matrix_difference_1_px_offset_job(port_input_name_1=edges[index],
                                                         port_input_name_2='THINNED_THR_THINNED_THR_' + dilated_edges_5[index] + '_FROM_L0',
                                                         port_output_name='ONLY_IN_' + edges[index],
                                                         level=CONFIG.PYRAMID_LEVEL.LEVEL_1)
        Application.do_matrix_difference_1_px_offset_job(port_input_name_1='THINNED_THR_THINNED_THR_' + dilated_edges_5[index] + '_FROM_L0',
                                                         port_input_name_2=edges[index],
                                                         port_output_name='ONLY_IN_' + dilated_edges_5[index] + '_DOWN_SAMPLED',
                                                         level=CONFIG.PYRAMID_LEVEL.LEVEL_1)
        list_to_save.append('ONLY_IN_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)
        list_to_save.append('ONLY_IN_' + dilated_edges_5[index] + '_DOWN_SAMPLED' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)

        ####################################################################################################################################
        # go with Canny dilated results down one level to compare. We want to see the differences
        ####################################################################################################################################
        Application.do_pyramid_level_down_job(port_input_name='CANNY_' + dilated_edges_5[index], number_of_lvl=1,
                                              port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                              port_output_name='CANNY_' + dilated_edges_5[index] + '_FROM_L0')

        # a line will result in min 255/4~60 pixel value in lowe levels
        Application.do_image_threshold_job(port_input_name='CANNY_' + dilated_edges_5[index] + '_FROM_L0',
                                           input_value=60,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THR_CANNY_' + dilated_edges_5[index] + '_FROM_L0',
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        # list_to_save.append('THR_CANNY_' + dilated_edges_5[index] + '_FROM_L0' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_thinning_guo_hall_image_job(port_input_name='THR_CANNY_' + dilated_edges_5[index] + '_FROM_L0',
                                                   port_output_name='THINNED_THR_CANNY_' + dilated_edges_5[index] + '_FROM_L0',
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        list_to_save.append('THINNED_THR_CANNY_' + dilated_edges_5[index] + '_FROM_L0' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)

        # Let's see the differences of images taking in consideration 1px offset in each direction because of the level manipulation
        Application.do_matrix_difference_1_px_offset_job(port_input_name_1='CANNY_' + edges[index],
                                                         port_input_name_2='THINNED_THR_CANNY_' + dilated_edges_5[index] + '_FROM_L0',
                                                         port_output_name='ONLY_IN_CANNY_' + edges[index],
                                                         level=CONFIG.PYRAMID_LEVEL.LEVEL_1)
        Application.do_matrix_difference_1_px_offset_job(port_input_name_1='THINNED_THR_CANNY_' + dilated_edges_5[index] + '_FROM_L0',
                                                         port_input_name_2='CANNY_' + edges[index],
                                                         port_output_name='ONLY_IN_CANNY_' + dilated_edges_5[index] + '_DOWN_SAMPLED',
                                                         level=CONFIG.PYRAMID_LEVEL.LEVEL_1)
        list_to_save.append('ONLY_IN_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)
        list_to_save.append('ONLY_IN_CANNY_' + dilated_edges_5[index] + '_DOWN_SAMPLED' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_1)

        ####################################################################################################################################
        # variant1 - Expanding the edge image from level 1 to level 0.
        ####################################################################################################################################
        ####################################################################################################################################
        # Gradient
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='THINNED_THR_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                            number_of_lvl=1)
        Application.do_image_threshold_job(port_input_name='EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                           port_output_name='THR_EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                           input_value=1,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY)
        Application.do_thinning_guo_hall_image_job(port_input_name='THR_EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                                   port_output_name='THINNED_THR_EXPANDED_FROM_L1_THINNED_THR_' + edges[index])
        list_to_save.append('THINNED_THR_EXPANDED_FROM_L1_THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('THINNED_THR_EXPANDED_FROM_L1_THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        ####################################################################################################################################
        # Canny
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='CANNY_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L1_CANNY_' + edges[index],
                                            number_of_lvl=1)
        Application.do_image_threshold_job(port_input_name='EXPANDED_FROM_L1_CANNY_' + edges[index],
                                           port_output_name='THRESHOLDED_EXPANDED_FROM_L1_CANNY_' + edges[index],
                                           input_value=1,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY)
        Application.do_thinning_guo_hall_image_job(port_input_name='THRESHOLDED_EXPANDED_FROM_L1_CANNY_' + edges[index],
                                                   port_output_name='THINNED_THRESHOLDED_EXPANDED_FROM_L1_CANNY_' + edges[index])
        list_to_save.append('THINNED_THRESHOLDED_EXPANDED_FROM_L1_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('THINNED_THRESHOLDED_EXPANDED_FROM_L1_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        ####################################################################################################################################
        # variant 2 Reconstructing the edge image from level 1 to level 0.
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='THINNED_THR_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                            number_of_lvl=1)

        Application.do_laplacian_pyramid_from_img_diff_job(port_input_name_1='GRAY_RAW',
                                                           port_input_name_2='EXPAND_GRAY_RAW',
                                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_matrix_bitwise_and_job(port_input_name_1='LAPLACE_PYRAMID_GRAY_RAW',
                                              port_input_name_2='EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                              port_output_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_thinning_guo_hall_image_job(port_input_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                                   port_output_name='THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L1_THINNED_THR_' + edges[
                                                       index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_save.append(
            'THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L1_THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append(
            'THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L1_THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        ####################################################################################################################################
        # Canny
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='CANNY_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L1_CANNY_' + edges[index],
                                            number_of_lvl=1)

        Application.do_laplacian_pyramid_from_img_diff_job(port_input_name_1='GRAY_RAW',
                                                           port_input_name_2='EXPAND_GRAY_RAW',
                                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_matrix_bitwise_and_job(port_input_name_1='LAPLACE_PYRAMID_GRAY_RAW',
                                              port_input_name_2='EXPANDED_FROM_L1_CANNY_' + edges[index],
                                              port_output_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L1_CANNY_' + edges[index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_thinning_guo_hall_image_job(port_input_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L1_CANNY_' + edges[index],
                                                   port_output_name='THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L1_CANNY_' + edges[index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_save.append('THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L1_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L1_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        ####################################################################################################################################
        # variant 3 Intersect the expanded edge maps from level 1 with edge map from level 0.
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='THINNED_THR_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                            number_of_lvl=1)

        Application.do_matrix_bitwise_and_job(port_input_name_1='THINNED_THR_' + edges[index],
                                              port_input_name_2='EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                              port_output_name='INTERSECTION_THINNED_THR_' + edges[
                                                  index] + '_AND_EXPANDED_FROM_L1_THINNED_THR_' + edges[index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_save.append('INTERSECTION_THINNED_THR_' + edges[index] + '_AND_EXPANDED_FROM_L1_THINNED_THR_' + edges[
            index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('INTERSECTION_THINNED_THR_' + edges[index] + '_AND_EXPANDED_FROM_L1_THINNED_THR_' + edges[
            index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        ####################################################################################################################################
        # Canny
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='CANNY_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L1_CANNY_' + edges[index],
                                            number_of_lvl=1)

        Application.do_matrix_bitwise_and_job(port_input_name_1='CANNY_' + edges[index],
                                              port_input_name_2='EXPANDED_FROM_L1_CANNY_' + edges[index],
                                              port_output_name='INTERSECTION_CANNY_' + edges[index] + '_AND_EXPANDED_FROM_L1_CANNY_' +
                                                               edges[index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_save.append(
            'INTERSECTION_CANNY_' + edges[index] + '_AND_EXPANDED_FROM_L1_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append(
            'INTERSECTION_CANNY_' + edges[index] + '_AND_EXPANDED_FROM_L1_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
    ########################################################################################################################################
    ########################################################################################################################################
    # FOR LVL2 -> LVL0
    ########################################################################################################################################
    ########################################################################################################################################
    # go down 2 levels
    Application.do_pyramid_level_down_job(port_input_name='GRAY_RAW', number_of_lvl=2,
                                          port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0)

    for index in range(len(edges)):
        ####################################################################################################################################
        # edge gradient
        ####################################################################################################################################
        # Apply Gradient 3x3 on L0
        Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', port_output_name='GAUSS_GREY_RAW',
                                               kernel_size=kernel, sigma=sigma,
                                               level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_first_order_derivative_operators(port_input_name='GAUSS_GREY_RAW',
                                                        operator=edges[index],
                                                        port_output_name=edges[index])

        Application.do_image_threshold_job(port_input_name=edges[index],
                                           input_value=60,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THR_' + edges[index],
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_thinning_guo_hall_image_job(port_input_name='THR_' + edges[index],
                                                   port_output_name='THINNED_THR_' + edges[index])

        list_to_save.append('THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        # Apply Gradient 7x7 on L0
        Application.do_first_order_derivative_operators(port_input_name='GAUSS_GREY_RAW',
                                                        operator=dilated_edges_7[index],
                                                        port_output_name=dilated_edges_7[index])

        Application.do_image_threshold_job(port_input_name=dilated_edges_7[index],
                                           input_value=60,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THR_' + dilated_edges_7[index],
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_thinning_guo_hall_image_job(port_input_name='THR_' + dilated_edges_7[index],
                                                   port_output_name='THINNED_THR_' + dilated_edges_7[index])

        list_to_save.append('THINNED_THR_' + dilated_edges_7[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('THINNED_THR_' + dilated_edges_7[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)

        # Apply Gradient 3x3 on L2
        Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', port_output_name='GAUSS_GREY_RAW',
                                               kernel_size=kernel, sigma=sigma,
                                               level=CONFIG.PYRAMID_LEVEL.LEVEL_2)

        Application.do_first_order_derivative_operators(port_input_name='GAUSS_GREY_RAW',
                                                        operator=edges[index],
                                                        port_output_name=edges[index],
                                                        level=CONFIG.PYRAMID_LEVEL.LEVEL_2)

        Application.do_image_threshold_job(port_input_name=edges[index],
                                           input_value=60,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THR_' + edges[index],
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_2)

        Application.do_thinning_guo_hall_image_job(port_input_name='THR_' + edges[index],
                                                   port_output_name='THINNED_THR_' + edges[index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_2)

        list_to_save.append('THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_2)
        ####################################################################################################################################
        # Canny
        ####################################################################################################################################
        # Apply canny normal on L0
        Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', port_output_name='GAUSS_GREY_RAW',
                                               kernel_size=kernel, sigma=sigma,
                                               level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_max_pixel_image_job(port_input_name='GAUSS_GREY_RAW', level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_canny_ratio_threshold_job(port_input_name='GRAY_RAW',
                                                 edge_detector=edges[index],
                                                 canny_config_value='MAX_PX_GAUSS_GREY_RAW',
                                                 port_output_name='CANNY_' + edges[index],
                                                 do_blur=True, kernel_blur_size=kernel, sigma=sigma)

        list_to_save.append('CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)

        # Apply canny with dilated sobel 7x7 on L0
        Application.do_canny_ratio_threshold_job(port_input_name='GRAY_RAW',
                                                 edge_detector=dilated_edges_7[index],
                                                 canny_config_value='MAX_PX_GAUSS_GREY_RAW',
                                                 port_output_name='CANNY_' + dilated_edges_7[index],
                                                 do_blur=True, kernel_blur_size=3, sigma=1)

        list_to_save.append('CANNY_' + dilated_edges_7[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('CANNY_' + dilated_edges_7[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)

        # Apply canny with sobel 3x3 on L2
        Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', port_output_name='GAUSS_GREY_RAW',
                                               kernel_size=kernel, sigma=sigma,
                                               level=CONFIG.PYRAMID_LEVEL.LEVEL_2)

        Application.do_max_pixel_image_job(port_input_name='GAUSS_GREY_RAW', level=CONFIG.PYRAMID_LEVEL.LEVEL_2)

        Application.do_canny_ratio_threshold_job(port_input_name='GRAY_RAW',
                                                 edge_detector=edges[index],
                                                 canny_config_value='MAX_PX_GAUSS_GREY_RAW',
                                                 level=CONFIG.PYRAMID_LEVEL.LEVEL_2,
                                                 port_output_name='CANNY_' + edges[index],
                                                 do_blur=True, kernel_blur_size=3, sigma=1)

        list_to_save.append('CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_2)
        ####################################################################################################################################
        # go with Canny dilated results down two levels to compare. We want to see the differences
        ####################################################################################################################################
        # ##################################################################################################################################
        # # go with gradient dilated results down one level to compare. We want to see the differences
        # ##################################################################################################################################
        Application.do_pyramid_level_down_job(port_input_name='THINNED_THR_' + dilated_edges_7[index], number_of_lvl=2,
                                              port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                              port_output_name='THINNED_THR_' + dilated_edges_7[index] + '_FROM_L0')
        list_to_save.append('THINNED_THR_' + dilated_edges_7[index] + '_FROM_L0' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_2)
        # a line will result in min 255/4~60 pixel value in lowe levels
        Application.do_image_threshold_job(port_input_name='THINNED_THR_' + dilated_edges_7[index] + '_FROM_L0',
                                           input_value=60,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THINNED_THR_THR_' + dilated_edges_7[index] + '_FROM_L0',
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_2)
        list_to_save.append('THINNED_THR_THR_' + dilated_edges_7[index] + '_FROM_L0' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_2)
        # Let's see the differences of images taking in consideration 1px offset in each direction because of the level manipulation
        Application.do_matrix_difference_1_px_offset_job(port_input_name_1=edges[index],
                                                         port_input_name_2='THINNED_THR_THR_' + dilated_edges_7[index] + '_FROM_L0',
                                                         port_output_name='ONLY_IN_' + edges[index],
                                                         level=CONFIG.PYRAMID_LEVEL.LEVEL_2)
        Application.do_matrix_difference_1_px_offset_job(port_input_name_1='THINNED_THR_THR_' + dilated_edges_7[index] + '_FROM_L0',
                                                         port_input_name_2=edges[index],
                                                         port_output_name='ONLY_IN_' + dilated_edges_7[index] + '_DOWN_SAMPLED',
                                                         level=CONFIG.PYRAMID_LEVEL.LEVEL_2)
        list_to_save.append('ONLY_IN_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_2)
        list_to_save.append('ONLY_IN_' + dilated_edges_7[index] + '_DOWN_SAMPLED' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_2)
        ####################################################################################################################################
        # go with Canny dilated results down one level to compare. We want to see the differences
        ####################################################################################################################################
        Application.do_pyramid_level_down_job(port_input_name='CANNY_' + dilated_edges_7[index], number_of_lvl=2,
                                              port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                              port_output_name='CANNY_' + dilated_edges_7[index] + '_FROM_L0')
        # a line will result in min 255/4~60 pixel value in lowe levels
        Application.do_image_threshold_job(port_input_name='CANNY_' + dilated_edges_7[index] + '_FROM_L0',
                                           input_value=15,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           port_output_name='THR_CANNY_' + dilated_edges_7[index] + '_FROM_L0',
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_2)
        # list_to_save.append('THR_CANNY_' + dilated_edges_7[index] + '_FROM_L0' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_2)

        Application.do_thinning_guo_hall_image_job(port_input_name='THR_CANNY_' + dilated_edges_7[index] + '_FROM_L0',
                                                   port_output_name='THINNED_THR_CANNY_' + dilated_edges_7[index] + '_FROM_L0',
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_2)

        # list_to_save.append('THINNED_THR_CANNY_' + dilated_edges_7[index] + '_FROM_L0' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_2)

        # Let's see the differences of images taking in consideration 1px offset in each direction because of the level manipulation
        Application.do_matrix_difference_1_px_offset_job(port_input_name_1='CANNY_' + edges[index],
                                                         port_input_name_2='THINNED_THR_CANNY_' + dilated_edges_7[index] + '_FROM_L0',
                                                         port_output_name='ONLY_IN_CANNY_' + edges[index],
                                                         level=CONFIG.PYRAMID_LEVEL.LEVEL_2)
        Application.do_matrix_difference_1_px_offset_job(port_input_name_1='THINNED_THR_CANNY_' + dilated_edges_7[index] + '_FROM_L0',
                                                         port_input_name_2='CANNY_' + edges[index],
                                                         port_output_name='ONLY_IN_CANNY_' + dilated_edges_7[index] + '_DOWN_SAMPLED',
                                                         level=CONFIG.PYRAMID_LEVEL.LEVEL_2)
        list_to_save.append('ONLY_IN_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_2)
        list_to_save.append('ONLY_IN_CANNY_' + dilated_edges_7[index] + '_DOWN_SAMPLED' + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_2)
        ####################################################################################################################################
        # variant 1 -  Expanding the edge image from level 1 to level 0.
        ####################################################################################################################################
        ####################################################################################################################################
        # Gradient
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='THINNED_THR_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_2,
                                            port_output_name='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                            number_of_lvl=1)
        Application.do_image_threshold_job(port_input_name='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                           port_output_name='THR_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                           input_value=1,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_1)
        Application.do_thinning_guo_hall_image_job(port_input_name='THR_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                                   port_output_name='THINNED_THR_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_pyramid_level_up_job(port_input_name='THINNED_THR_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                            number_of_lvl=1)
        Application.do_image_threshold_job(port_input_name='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                           port_output_name='THR_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                           input_value=1,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        Application.do_thinning_guo_hall_image_job(port_input_name='THR_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                                   port_output_name='THINNED_THR_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_save.append('THINNED_THR_EXPANDED_FROM_L2_THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('THINNED_THR_EXPANDED_FROM_L2_THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        ####################################################################################################################################
        # Canny
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='CANNY_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_2,
                                            port_output_name='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                            number_of_lvl=1)
        Application.do_image_threshold_job(port_input_name='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                           port_output_name='THR_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                           input_value=1,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_1)
        Application.do_thinning_guo_hall_image_job(port_input_name='THR_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                                   port_output_name='THINNING_THR_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_pyramid_level_up_job(port_input_name='THINNING_THR_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                            number_of_lvl=1)
        Application.do_image_threshold_job(port_input_name='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                           port_output_name='THR_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                           input_value=1,
                                           input_threshold_type=CONFIG.THRESHOLD_CONFIG.THR_BINARY,
                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        Application.do_thinning_guo_hall_image_job(port_input_name='THR_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                                   port_output_name='THINNED_THR_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_save.append('THINNED_THR_EXPANDED_FROM_L2_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('THINNED_THR_EXPANDED_FROM_L2_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        ####################################################################################################################################
        # variant 2 Reconstructing the edge image from level 1 to level 0.
        ####################################################################################################################################
        ####################################################################################################################################
        # Gradient
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='THINNED_THR_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_2,
                                            port_output_name='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                            number_of_lvl=1)
        # obtain laplace pyramid for L2
        Application.do_pyramid_level_up_job(port_input_name='GRAY_RAW',
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_2,
                                            port_output_name='EXPAND_GRAY_RAW',
                                            number_of_lvl=1)

        Application.do_laplacian_pyramid_from_img_diff_job(port_input_name_1='GRAY_RAW',
                                                           port_input_name_2='EXPAND_GRAY_RAW',
                                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_matrix_bitwise_and_job(port_input_name_1='LAPLACE_PYRAMID_GRAY_RAW',
                                              port_input_name_2='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                              port_output_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_thinning_guo_hall_image_job(port_input_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                                   port_output_name='THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L2_THINNED_THR_' + edges[
                                                       index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_pyramid_level_up_job(port_input_name='THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                            number_of_lvl=1)

        Application.do_laplacian_pyramid_from_img_diff_job(port_input_name_1='GRAY_RAW',
                                                           port_input_name_2='EXPAND_GRAY_RAW',
                                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_matrix_bitwise_and_job(port_input_name_1='LAPLACE_PYRAMID_GRAY_RAW',
                                              port_input_name_2='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                              port_output_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        #
        Application.do_thinning_guo_hall_image_job(port_input_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                                   port_output_name='THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L2_THINNED_THR_' + edges[
                                                       index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_save.append(
            'THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L2_THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append(
            'THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L2_THINNED_THR_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        ####################################################################################################################################
        # Canny
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='CANNY_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_2,
                                            port_output_name='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                            number_of_lvl=1)
        # obtain laplace pyramid for L2
        Application.do_pyramid_level_up_job(port_input_name='GRAY_RAW',
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_2,
                                            port_output_name='EXPAND_GRAY_RAW',
                                            number_of_lvl=1)

        Application.do_laplacian_pyramid_from_img_diff_job(port_input_name_1='GRAY_RAW',
                                                           port_input_name_2='EXPAND_GRAY_RAW',
                                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_matrix_bitwise_and_job(port_input_name_1='LAPLACE_PYRAMID_GRAY_RAW',
                                              port_input_name_2='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                              port_output_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_thinning_guo_hall_image_job(port_input_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                                   port_output_name='THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_pyramid_level_up_job(port_input_name='THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                            number_of_lvl=1)

        Application.do_laplacian_pyramid_from_img_diff_job(port_input_name_1='GRAY_RAW',
                                                           port_input_name_2='EXPAND_GRAY_RAW',
                                                           level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

        Application.do_matrix_bitwise_and_job(port_input_name_1='LAPLACE_PYRAMID_GRAY_RAW',
                                              port_input_name_2='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                              port_output_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        #
        Application.do_thinning_guo_hall_image_job(port_input_name='INTERSECTION_LAPLACE_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                                   port_output_name='THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L2_CANNY_' + edges[index],
                                                   level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_save.append('THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L2_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('THINNED_INTERSECTION_LAPLACE_EXPANDED_FROM_L2_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        ####################################################################################################################################
        # variant a 3 Intersect the expanded edge maps from level 1 with edge map from level 0.
        ####################################################################################################################################
        ####################################################################################################################################
        # edge gradient
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='THINNED_THR_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_2,
                                            port_output_name='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                            number_of_lvl=1)

        Application.do_matrix_bitwise_and_job(port_input_name_1='THINNED_THR_' + edges[index],
                                              port_input_name_2='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                              port_output_name='INTERSECTION_THINNED_THR_' + edges[
                                                  index] + '_AND_EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_pyramid_level_up_job(port_input_name='THINNED_THR_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                            number_of_lvl=1)

        Application.do_matrix_bitwise_and_job(port_input_name_1='THINNED_THR_' + edges[index],
                                              port_input_name_2='EXPANDED_FROM_L2_THINNED_THR_' + edges[index],
                                              port_output_name='INTERSECTION_THINNED_THR_' + edges[
                                                  index] + '_AND_EXPANDED_FROM_L2_THINNED_THR_' + edges[
                                                                   index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_save.append('INTERSECTION_THINNED_THR_' + edges[index] + '_AND_EXPANDED_FROM_L2_THINNED_THR_' + edges[
            index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append('INTERSECTION_THINNED_THR_' + edges[index] + '_AND_EXPANDED_FROM_L2_THINNED_THR_' + edges[
            index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        ####################################################################################################################################
        # Canny
        ####################################################################################################################################
        Application.do_pyramid_level_up_job(port_input_name='CANNY_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_2,
                                            port_output_name='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                            number_of_lvl=1)

        Application.do_matrix_bitwise_and_job(port_input_name_1='CANNY_' + edges[index],
                                              port_input_name_2='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                              port_output_name='INTERSECTION_CANNY_' + edges[index] + '_AND_EXPANDED_FROM_L2_CANNY_' +
                                                               edges[index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_1)

        Application.do_pyramid_level_up_job(port_input_name='CANNY_' + edges[index],
                                            port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                            port_output_name='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                            number_of_lvl=1)

        Application.do_matrix_bitwise_and_job(port_input_name_1='CANNY_' + edges[index],
                                              port_input_name_2='EXPANDED_FROM_L2_CANNY_' + edges[index],
                                              port_output_name='INTERSECTION_CANNY_' + edges[index] + '_AND_EXPANDED_FROM_L2_CANNY_' +
                                                               edges[
                                                                   index],
                                              level=CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_save.append(
            'INTERSECTION_CANNY_' + edges[index] + '_AND_EXPANDED_FROM_L2_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)
        list_to_benchmark.append(
            'INTERSECTION_CANNY_' + edges[index] + '_AND_EXPANDED_FROM_L2_CANNY_' + edges[index] + '_' + CONFIG.PYRAMID_LEVEL.LEVEL_0)

    Application.create_config_file()

    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save=list_to_save)

    # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=0)

    Application.run_application()

    Benchmarking.run_FOM_benchmark(input_location='Logs/application_results/',
                                   gt_location='TestData/dilation_test/validate/',
                                   raw_image='TestData/dilation_test/test/',
                                   jobs_set=list_to_benchmark)

    Benchmarking.run_FOM_benchmark(input_location='Logs/application_results/',
                                   gt_location='TestData/BSR/BSDS500/data/groundTruth/img',
                                   raw_image='TestData/BSR/BSDS500/data/images/buildings',
                                   jobs_set=list_to_benchmark)

    Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
                                                gt_location='TestData/BSR/BSDS500/data/groundTruth/all',
                                                raw_image='TestData/BSR/BSDS500/data/images/all',
                                                jobs_set=list_to_benchmark)
    Utils.close_files()


if __name__ == "__main__":
    main()
