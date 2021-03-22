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
    Application.set_input_image_folder('TestData/dilation_test/test')

    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='RAW_GRAY')

    first_order_edge = [
        CONFIG.FILTERS.SOBEL_3x3
        # , CONFIG.FILTERS.ORHEI_3x3
    ]

    list = []

    for edge in first_order_edge:
        for kernel_gaus in [3, 5, 7, 9, 11]:
            for grad_thr in range(5,170,5):
                for anc_thr in range(5,50,5):
                    for sc_int in range(1, kernel_gaus, 2):
                        blur = Application.do_gaussian_blur_image_job(port_input_name='RAW_GRAY', kernel_size=kernel_gaus, sigma=0)
                        e3, e4 = Application.do_edge_drawing_mod_job(port_input_name=blur, operator=edge,
                                                                     gradient_thr=grad_thr, anchor_thr=anc_thr, scan_interval=sc_int,
                                                                     max_edges=10, max_points_edge=10)
                        list.append(e3 + '_L0')

    Application.create_config_file(verbose=False)
    Application.configure_save_pictures(location='DEFAULT', job_name_in_port=True, ports_to_save='ALL')
    # Application.configure_show_pictures(ports_to_show=list_to_save, time_to_show=200)

    # Application.run_application()
    # Do bsds benchmarking
    # Be ware not to activate job_name_in_port in Application.configure_save_pictures

    # Benchmarking.run_FOM_benchmark(input_location='Logs/application_results',
    #                                gt_location='TestData/dilation_test/validate',
    #                                raw_image='TestData/dilation_test/test',
    #                                jobs_set=list)

    Utils.create_latex_fom_table()


if __name__ == "__main__":
    main()
