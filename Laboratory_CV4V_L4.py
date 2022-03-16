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


def main_basic_transform():
    """

    """
    # Delete output folder for new exectution
    Application.delete_folder_appl_out()
    # Set location of input images
    Application.set_input_image_folder('TestData\Laboratory_CV4V\Lab_4')
    # Set job in framework to get image, uses cv2
    original_img = Application.do_get_image_job()

    # Blur image
    for i in range(0, 360, 45):
        rotate_image = Application.do_rotate_image_job(port_input_name=original_img, angle=i, reshape=True, is_rgb=True)

    rotate_image = Application.do_flip_image_job(port_input_name=original_img, flip_horizontal=False,
                                                 flip_vertical=True, is_rgb=True)

    # Difference of original - rotate
    # diff_1 = Application.do_matrix_difference_job(port_input_name_1=original_img, port_input_name_2=rotate_image)

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
    main_basic_transform()