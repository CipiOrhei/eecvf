import Application
# noinspection PyUnresolvedReferences
import Benchmarking
# noinspection PyUnresolvedReferences
import MachineLearning
# noinspection PyUnresolvedReferences
import config_main as CONFIG
# noinspection PyUnresolvedReferences
import Utils

def main():
    """

    """
    Application.delete_folder_appl_out()
    Application.set_input_image_folder('TestData/smoke_test')
    raw = Application.do_get_image_job('RAW')
    grey = Application.do_grayscale_transform_job(port_input_name='RAW')
    hist_eq = Application.do_histogram_equalization_job(port_input_name=grey, save_histogram=False)
    sharpen = Application.do_sharpen_filter_job(port_input_name=grey, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1)
    sharpen_2 = Application.do_sharpen_filter_job(port_input_name=grey, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2)
    sharpen_dil = Application.do_sharpen_filter_job(port_input_name=grey, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1)
    um_dil = Application.do_unsharp_filter_job(port_input_name=grey, is_rgb=False)
    um_dil = Application.do_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strenght=1)
    um_dil = Application.do_unsharp_filter_expanded_job(port_input_name=grey, is_rgb=False, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, strenght=1)

    sharpen = Application.do_sharpen_filter_job(port_input_name='RAW', is_rgb=True, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1)
    sharpen_2 = Application.do_sharpen_filter_job(port_input_name='RAW', is_rgb=True, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_2)
    sharpen_dil = Application.do_sharpen_filter_job(port_input_name='RAW', is_rgb=True, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1)
    um_dil = Application.do_unsharp_filter_job(port_input_name='RAW', is_rgb=True)
    um_dil = Application.do_unsharp_filter_expanded_job(port_input_name='RAW', is_rgb=True, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_1, strenght=1)
    um_dil = Application.do_unsharp_filter_expanded_job(port_input_name='RAW', is_rgb=True, kernel=CONFIG.FILTERS_SECOND_ORDER.LAPLACE_DILATED_5x5_1, strenght=1)

    # Application.do_histogram_job(port_input_name=grey)
    # Application.do_histogram_job(port_input_name=hist_eq)
    # Application.do_histogram_job(port_input_name=sharpen)
    # Application.do_histogram_job(port_input_name=sharpen_2)
    # Application.do_histogram_job(port_input_name=sharpen_dil)
    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save='ALL', job_name_in_port=True)
    # Application.configure_show_pictures(ports_to_show='ALL', time_to_show=500, to_rotate=False)
    Application.run_application()

    Utils.close_files()


if __name__ == "__main__":
    main()
