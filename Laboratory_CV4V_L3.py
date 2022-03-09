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



def main_basic_filters():
    """

    """
    # Delete output folder for new exectution
    Application.delete_folder_appl_out()
    # Set location of input images
    Application.set_input_image_folder('TestData/smoke_test')
    # Set job in framework to get image, uses cv2
    original_img = Application.do_get_image_job()

    # Blur image
    blured_image = Application.do_mean_blur_job(port_input_name=original_img, kernel_size=3, is_rgb=True)

    # Sharpening image
    sharped_image = Application.do_sharpen_filter_job(port_input_name=original_img, kernel=3, is_rgb=True)

    # Difference of original - blur
    diff_1 = Application.do_matrix_difference_job(port_input_name_1=original_img, port_input_name_2=blured_image)
    # Difference of original - blur
    diff_2 = Application.do_matrix_difference_job(port_input_name_1=sharped_image, port_input_name_2=original_img)


    # create config file of jobs
    Application.create_config_file()
    # config which ports to save, should get a list with pyramid level at end
    Application.configure_save_pictures(ports_to_save='ALL')
    # config which ports to show, should get a list with pyramid level at end
    Application.configure_show_pictures(ports_to_show='ALL', time_to_show=0, to_rotate=False)
    # Start application
    Application.run_application()
    # Close all external resources of application
    Utils.close_files()


def main_basic_edge():
    """

    """
    # Delete output folder for new exectution
    Application.delete_folder_appl_out()
    # Set location of input images
    Application.set_input_image_folder('TestData/smoke_test')
    # Set job in framework to get image, uses cv2
    original_img = Application.do_get_image_job()

    # Blur image
    blured_image = Application.do_mean_blur_job(port_input_name=original_img, kernel_size=3, is_rgb=True)

    Application.do_first_order_derivative_operators(port_input_name=original_img, operator=CONFIG.FILTERS.SOBEL_3x3, is_rgb=True)
    Application.do_first_order_derivative_operators(port_input_name=original_img, operator=CONFIG.FILTERS.KIRSCH_3x3, is_rgb=True)

    # Application.do_first_order_derivative_operators(port_input_name=blured_image, operator=CONFIG.FILTERS.SOBEL_3x3, is_rgb=True)
    # Application.do_first_order_derivative_operators(port_input_name=blured_image, operator=CONFIG.FILTERS.KIRSCH_3x3, is_rgb=True)

    # create config file of jobs
    Application.create_config_file()
    # config which ports to save, should get a list with pyramid level at end
    Application.configure_save_pictures(ports_to_save='ALL')
    # config which ports to show, should get a list with pyramid level at end
    Application.configure_show_pictures(ports_to_show='ALL', time_to_show=0, to_rotate=False)
    # Start application
    Application.run_application()
    # Close all external resources of application
    Utils.close_files()


if __name__ == "__main__":
    # main_basic_filters()
    main_basic_edge()