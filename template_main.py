import Application
import Benchmarking
import MachineLearning
import config_main as CONFIG
import Utils


def main():
    """

    """
    Application.delete_folder_appl_out()
    Application.set_input_image_folder('TestData/smoke_test')
    Application.do_get_image_job('RAW')
    Application.set_number_waves(2)
    grey = Application.do_grayscale_transform_job(port_input_name='RAW')

    Application.do_max_pixel_image_job(port_input_name=grey, level=CONFIG.PYRAMID_LEVEL.LEVEL_0)

    Application.do_pyramid_level_down_job(port_input_name=grey, number_of_lvl=1)
    Application.do_pyramid_level_down_job(port_input_name='RAW', number_of_lvl=1, is_rgb=True)

    Application.do_pyramid_level_up_job(port_input_name=grey, port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                        number_of_lvl=1, is_rgb=False)
    Application.do_pyramid_level_up_job(port_input_name='RAW', port_input_lvl=CONFIG.PYRAMID_LEVEL.LEVEL_1,
                                        number_of_lvl=1, is_rgb=True)

    xt = Application.do_dob_job(port_input_name=grey,
                                is_rgb=False)
    print(xt)
    Application.do_edge_label_job(port_input_name=xt, connectivity=4)
    Application.do_edge_label_job(port_input_name=xt, connectivity=8)

    Application.do_hough_lines_job(port_input_name=xt, vote_threshold=150)

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL')
    Application.configure_show_pictures(ports_to_show='ALL', time_to_show=500, to_rotate=False)
    Application.run_application()

    Utils.close_files()


if __name__ == "__main__":
    main()
