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


def main():
    """
    Main function of framework Please look in example_main for all functions you can use
    """

    folders = [
        r'D:\Repos\eecvf_github\TestData\smoke_test',
        r'D:\Repos\eecvf_github\TestData\dilation_test\validate'
    ]

    for folder in folders:
        Application.set_input_image_folder(folder)
        Application.set_output_image_folder('D:\\Repos\\eecvf_github\\Logs\\' + folder.split('\\')[-1])
        Application.delete_folder_appl_out()

        grey = Application.do_get_image_job(port_output_name='GRAY_RAW', direct_grey=True)

        Application.create_config_file()
        Application.configure_save_pictures(ports_to_save='ALL')
        # Application.configure_show_pictures(ports_to_show=list, time_to_show=0)

        Application.run_application()

    Utils.close_files()


if __name__ == "__main__":
    main()