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
This module contains the code used for the following paper:
  title={A Novel Edge Detection Operator for Identifying Buildings in Augmented Reality Applications},
  author={Orhei, Ciprian and Vert, Silviu and Vasiu, Radu},
  booktitle={International Conference on Information and Software Technologies},
  pages={208--219},
  year={2020},
  organization={Springer}
"""


def main():
    """
    Main function of framework Please look in example_main for all functions
    you can use
    """
    Application.set_input_image_folder('TestData/SpotlightHeritage')
    Application.delete_folder_appl_out()

    Application.do_get_image_job(port_output_name='RAW')

    Application.do_grayscale_transform_job(port_input_name='RAW',
                                           port_output_name='GRAY_RAW')

    edges = [
        CONFIG.FILTERS.SOBEL_3x3, CONFIG.FILTERS.SOBEL_5x5
        , CONFIG.FILTERS.PREWITT_3x3, CONFIG.FILTERS.PREWITT_5x5
        , CONFIG.FILTERS.SCHARR_3x3, CONFIG.FILTERS.SCHARR_5x5
        , CONFIG.FILTERS.ORHEI_3x3, CONFIG.FILTERS.ORHEI_B_5x5, CONFIG.FILTERS.ORHEI_5x5
    ]
    gaussian_kernel_size = 9
    sigma_kernel_size = 1.4

    Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW',
                                           kernel_size=gaussian_kernel_size,
                                           sigma=sigma_kernel_size)

    for edge in edges:
        Application.do_first_order_derivative_operators(port_input_name='GAUSS_BLUR_K_9_S_1_4_GRAY_RAW',
                                                        operator=edge)

        Application.do_canny_otsu_median_sigma_job(port_input_name='GAUSS_BLUR_K_9_S_1_4_GRAY_RAW',
                                                   edge_detector=edge, do_blur=False)

    Application.do_matrix_difference_job(
        port_input_name_1='CANNY_OTSU_MEDIAN_SIGMA_SOBEL_3x3_GAUSS_BLUR_K_9_S_1_4_GRAY_RAW',
        port_input_name_2='CANNY_OTSU_MEDIAN_SIGMA_ORHEI_3x3_GAUSS_BLUR_K_9_S_1_4_GRAY_RAW',
        port_output_name='ONLY_IN_CANNY_SOBEL_3x3')

    Application.do_matrix_difference_job(
        port_input_name_1='CANNY_OTSU_MEDIAN_SIGMA_ORHEI_3x3_GAUSS_BLUR_K_9_S_1_4_GRAY_RAW',
        port_input_name_2='CANNY_OTSU_MEDIAN_SIGMA_SOBEL_3x3_GAUSS_BLUR_K_9_S_1_4_GRAY_RAW',
        port_output_name='ONLY_IN_CANNY_PROPOSED_3x3')

    Application.do_edge_label_job(port_input_name='ONLY_IN_CANNY_SOBEL_3x3')
    Application.do_edge_label_job(port_input_name='ONLY_IN_CANNY_PROPOSED_3x3')

    Application.create_config_file()

    list_to_save = Application.create_list_ports_with_word('CANNY')
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save=list_to_save)
    # Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save='ALL')
    # Application.configure_show_pictures(time_to_show=150, ports_to_show='ALL')

    Application.run_application()

    Utils.close_files()
    Utils.plot_custom_list(port_list=['Nr Edge px ONLY_IN_CANNY_PROPOSED_3x3_L0', 'Nr Edge px ONLY_IN_CANNY_SOBEL_3x3_L0'],
                           name_to_save='Number of edge pixels', y_plot_name='Number of edge pixels',
                           show_plot=False, save_plot=True)

    Utils.plot_custom_list(port_list=['Nr Edges ONLY_IN_CANNY_PROPOSED_3x3_L0', 'Nr Edges ONLY_IN_CANNY_SOBEL_3x3_L0'],
                           name_to_save='Number of edges', y_plot_name='Number of edges',
                           show_plot=False, save_plot=True)

    Utils.plot_custom_list(port_list=['AVG px/edge ONLY_IN_CANNY_PROPOSED_3x3_L0', 'AVG px/edge ONLY_IN_CANNY_SOBEL_3x3_L0'],
                           name_to_save='Average number of pixel per edge', y_plot_name='Avg px per edge',
                           show_plot=False, save_plot=True)


if __name__ == "__main__":
    main()
