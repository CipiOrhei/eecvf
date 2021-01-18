# noinspection PyUnresolvedReferences
import Application
# noinspection PyUnresolvedReferences
import Benchmarking
# noinspection PyPep8Naming
import config_main as CONFIG

"""
Code for paper:
title={Custom Dilated Edge Detection Filters},
author={Bogdan, V and Bonchis, C and Orhei, C},
booktitle={International Conference in Central Europe on Computer Graphics, Visualization and Computer Vision},
pages={to appear},
publisher={V{\'a}clav Skala-UNION Agency}
year={May 2020},
organization={WSCG}
eprint={arXiv:1910.00138}
"""


def main():
    """Main function of framework Please look in example_main for all functions
    you can use
    """
    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/train')

    Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='RAW_GRAY')

    # Application.do_pyramid_level_down_job(port_name_input='RAW_GRAY', port_input_lvl=Application.PYRAMID_LEVEL.LEVEL_0,
    #                                  port_name_output='RAW_GRAY', number_of_lvl=4, verbose=False)

    levels = [CONFIG.PYRAMID_LEVEL.LEVEL_0]
    input_image_list = []

    for pyramid_level in levels:
        kernel_size_list = [3]
        sigma_list = [1.0]

        for kernel in kernel_size_list:
            for sigma in sigma_list:
                output_name = 'GAUS_BLUR' + '_' + str(kernel) + '_SIGMA_' + str(sigma).replace('.', '_')
                Application.do_gaussian_blur_image_job(port_input_name='RAW_GRAY', port_output_name=output_name,
                                                       kernel_size=kernel, sigma=sigma, level=pyramid_level)
                input_image_list.append(output_name)

                for image in input_image_list:
                    Application.do_max_pixel_image_job(port_input_name=image, port_output_name='MAX_' + image, level=pyramid_level)

                    first_order_derivatives_filters = [CONFIG.FILTERS.SOBEL_3x3, CONFIG.FILTERS.SOBEL_5x5, CONFIG.FILTERS.SOBEL_7x7,
                                                       CONFIG.FILTERS.SOBEL_DILATED_5x5, CONFIG.FILTERS.SOBEL_DILATED_7x7,
                                                       CONFIG.FILTERS.PREWITT_3x3, CONFIG.FILTERS.PREWITT_5x5,
                                                       CONFIG.FILTERS.PREWITT_DILATED_7x7,
                                                       CONFIG.FILTERS.PREWITT_DILATED_5x5, CONFIG.FILTERS.PREWITT_7x7,
                                                       CONFIG.FILTERS.SCHARR_3x3, CONFIG.FILTERS.SCHARR_5x5,
                                                       CONFIG.FILTERS.SCHARR_DILATED_5x5]

                    for edge in first_order_derivatives_filters:
                        Application.do_first_order_derivative_operators(port_input_name=image, operator=edge, level=pyramid_level)
                        thresholds = [60, 120]
                        for th in thresholds:
                            Application.do_image_threshold_job(port_input_name=edge + '_GAUS_BLUR_3_SIGMA_1_0',
                                                               input_value=th,
                                                               input_threshold_type='cv2.THRESH_BINARY',
                                                               port_output_name=edge + '_GAUS_BLUR_3_SIGMA_1_0_T_' + str(th),
                                                               level=pyramid_level)

                        Application.do_canny_ratio_threshold_job(port_input_name=image, edge_detector=edge,
                                                                 canny_config_value='MAX_' + image,
                                                                 port_output_name='CANNY_RATIO_TRH_' + edge, level=pyramid_level,
                                                                 do_blur=False, kernel_blur_size=kernel, sigma=sigma)

    Application.create_config_file(verbose=False)

    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save='ALL')

    # Application.configure_show_pictures(ports_to_show=list_to_save, time_to_show=200)

    Application.run_application()
    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures
    # Benchmarking.run_bsds500_boundry_benchmark(input_location='Logs/out',
    #                                            gt_location='TestData/BSR/BSDS500/data/groundTruth/test',
    #                                            raw_image='TestData/BSR/BSDS500/data/images/test',
    #                                            jobs_set=list_to_save)


if __name__ == "__main__":
    main()
