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
This module is an example to use the EECVF with video input stream.
"""


def main():
    """
    Main function of framework
    Please look in example_main for all functions you can use
    """
    Application.set_input_video('TestData/smoke_movies/car_camera_movie.mp4')

    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.do_get_video_job(port_output_name='RAW')

    Application.do_pyramid_level_down_job(port_input_name='RAW',
                                          port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_0,
                                          port_output_name='RAW',
                                          number_of_lvl=2, is_rgb=True)

    level = CONFIG.PYRAMID_LEVEL.LEVEL_2

    Application.do_grayscale_transform_job(port_input_name='RAW',
                                           port_output_name='RAW_GRAY',
                                           level=level)

    Application.do_median_pixel_image_job(port_input_name='RAW_GRAY',
                                          port_output_name='MEDIAN_RAW_GRAY',
                                          level=level)

    Application.do_canny_otsu_median_sigma_job(port_input_name='RAW_GRAY',
                                               edge_detector=CONFIG.FILTERS.ORHEI_3x3,
                                               port_output_name='CANNY_OTSU_MEDIAN_SIGMA_RAW_GRAY',
                                               level=level,
                                               do_blur=True,
                                               kernel_blur_size=9,
                                               sigma=1.4)

    Application.do_edge_label_job(port_input_name='CANNY_OTSU_MEDIAN_SIGMA_RAW_GRAY',
                                  port_output_name='EDGE_LABELED_CANNY_OTSU_MEDIAN_SIGMA_RAW_GRAY',
                                  level=level)

    # create config file for application
    Application.create_config_file()

    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=False, ports_to_save='ALL')

    list_to_show = [
        'CANNY_OTSU_MEDIAN_SIGMA_RAW_GRAY_' + str(level)
        , 'EDGE_LABELED_CANNY_OTSU_MEDIAN_SIGMA_RAW_GRAY_' + str(level)
        , 'OTSU_RAW_GRAY_GAUS_BLUR_K_9_S_1_4_IMG_' + str(level)
        , 'RAW_' + str(level)
    ]

    Application.configure_show_pictures(ports_to_show=['RAW_L0'], time_to_show=1)

    Application.run_application()

    Utils.close_files()


if __name__ == "__main__":
    main()
