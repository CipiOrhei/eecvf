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

def main_sablon():
    """
    Main function of framework Please look in example_main for all functions
    you can use
    """
    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/test')
    Application.set_output_image_folder('Logs/appl_temp')
    Application.delete_folder_appl_out()
    Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GRAY_RAW')

    list = []

    first_order_edge = [
        CONFIG.FILTERS.SOBEL_3x3
        , CONFIG.FILTERS.PREWITT_3x3
        , CONFIG.FILTERS.KIRSCH_3x3
        , CONFIG.FILTERS.KITCHEN_MALIN_3x3
        , CONFIG.FILTERS.KAYYALI_3x3
        , CONFIG.FILTERS.SCHARR_3x3
        , CONFIG.FILTERS.KROON_3x3
        , CONFIG.FILTERS.ORHEI_3x3
    ]

    for edge in first_order_edge:
        for kernel_gaus in [9]:
            for grad_thr in [50]:
                for anc_thr in [10]:
                    for sc_int in [1]:
                        blur = Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', kernel_size=kernel_gaus, sigma=0)
                        e3, e4 = Application.do_edge_drawing_mod_job(port_input_name=blur, operator=edge,
                                                                     gradient_thr=grad_thr, anchor_thr=anc_thr, scan_interval=sc_int,
                                                                     max_edges=100, max_points_edge=100)
                        list.append(e3 + '_L0')


    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=list, job_name_in_port=True)
    # Application.configure_show_pictures(ports_to_show=list, time_to_show=0)

    Application.run_application()

    # Benchmarking.run_FOM_benchmark(input_location='Logs/appl_temp/',
    #                                gt_location='TestData/BSR/BSDS500/data/groundTruth/test_img',
    #                                raw_image='TestData/BSR/BSDS500/data/images/test',
    #                                jobs_set=list)
    # Benchmarking.run_PSNR_benchmark(input_location='Logs/appl_temp/',
    #                                gt_location='TestData/BSR/BSDS500/data/groundTruth/test_img',
    #                                raw_image='TestData/BSR/BSDS500/data/images/test',
    #                                jobs_set=list)
    Utils.close_files()

def main():
    """
    Main function of framework Please look in example_main for all functions
    you can use
    """
    Application.set_input_image_folder('TestData/BSR/BSDS500/data/images/test')
    Application.set_output_image_folder('Logs/appl_temp_2')
    # Application.delete_folder_appl_out()
    # Benchmarking.delete_folder_benchmark_out()

    Application.do_get_image_job(port_output_name='RAW')
    Application.do_grayscale_transform_job(port_input_name='RAW', port_output_name='GRAY_RAW')

    smoothing_list = list()
    edge_to_evaluate = list()

    # for kernel_size in [3, 5]:
    #         median_output_name = 'MEDIAN_K_' + str(kernel_size)
    #         Application.do_median_blur_job(port_input_name='GRAY_RAW', kernel_size=kernel_size, port_output_name=median_output_name)
    #         smoothing_list.append(median_output_name)

    for kernel_size in [9]:
            median_output_name = 'GAUSS_K_' + str(kernel_size)
            Application.do_gaussian_blur_image_job(port_input_name='GRAY_RAW', kernel_size=kernel_size, sigma=0, port_output_name=median_output_name)
            smoothing_list.append(median_output_name)

    first_order_edge = [
        CONFIG.FILTERS.SOBEL_3x3
        , CONFIG.FILTERS.PREWITT_3x3
        , CONFIG.FILTERS.KIRSCH_3x3
        , CONFIG.FILTERS.KITCHEN_MALIN_3x3
        , CONFIG.FILTERS.KAYYALI_3x3
        , CONFIG.FILTERS.SCHARR_3x3
        , CONFIG.FILTERS.KROON_3x3
        , CONFIG.FILTERS.ORHEI_3x3
    ]
    for edge in first_order_edge:
        for blur in smoothing_list:
            for gt in [500]:
                gt = gt / 1000
                for at in [433]:
                    at = (1000-at) / 1000
                    for n in [2]:
                        Application.do_multi_otsu_job(port_input_name=blur, number_of_classes=n,
                                                      port_output_name='OTSU_MULTI_LEVEL_' + str(n) + '_' + blur)

                        grad_th = Application.do_value_manipulation_job(
                            terms_input_list=['OTSU_MULTI_LEVEL_' + str(n) + '_' + blur + '_VALUE_1', gt],
                            port_input_wave_list=[0, ''],
                            port_input_level_list=[CONFIG.PYRAMID_LEVEL.LEVEL_0, ''],
                            operation_list=['*'],
                            port_output='GRAD_THR_V_' + blur)

                        anchor_th = Application.do_value_manipulation_job(
                            # terms_input_list=[grad_th, at],
                            terms_input_list=['OTSU_MULTI_LEVEL_' + str(n) + '_' + blur + '_VALUE_1', at, grad_th],
                            port_input_wave_list=[0, '', 0],
                            port_input_level_list=[CONFIG.PYRAMID_LEVEL.LEVEL_0, '', CONFIG.PYRAMID_LEVEL.LEVEL_0],
                            operation_list=['*', '-'],
                            port_output='ANC_THR_V_' + blur)

                        edge_map_clasic, edge_seg_clasic = Application.do_edge_drawing_mod_job(port_input_name=blur, operator=edge,
                                                                                               gradient_thr=grad_th, anchor_thr=anchor_th, scan_interval=1,
                                                                                               port_edge_map_name_output='ED_' + str(n) + '_' + edge +
                                                                                                                         '_' + str(gt).replace('.', '_') +
                                                                                                                         '_' + str(at).replace('.', '_') + '_' + blur,
                                                                                               port_edges_name_output='ED_S_' + str(n) + '_' + edge + '_' +
                                                                                                                      str(gt).replace('.', '_') + '_' +
                                                                                                                      str(at).replace('.', '_') + '_' + blur,
                                                                                               max_edges=1, max_points_edge=1)
                        edge_to_evaluate.append(edge_map_clasic + '_L0')

    Application.create_config_file()
    Application.configure_save_pictures(ports_to_save=edge_to_evaluate, job_name_in_port=True)

    Application.run_application()

    # Benchmarking.run_FOM_benchmark(input_location='Logs/application_results',
    #                                gt_location='Logs/appl_temp/EDGE_DRAWING_MOD_THR_50_ANC_THR_10_SCAN_1_SOBEL_3x3_GAUSS_BLUR_K_9_S_0_GRAY_RAW_L0',
    #                                # gt_location='TestData/BSR/BSDS500/data/groundTruth/test_img',
    #                                raw_image='TestData/BSR/BSDS500/data/images/test',
    #                                jobs_set=edge_to_evaluate)
    # Benchmarking.run_PSNR_benchmark(input_location='Logs/application_results',
    #                                gt_location='Logs/appl_temp/EDGE_DRAWING_MOD_THR_50_ANC_THR_10_SCAN_1_SOBEL_3x3_GAUSS_BLUR_K_9_S_0_GRAY_RAW_L0',
    #                                raw_image='TestData/BSR/BSDS500/data/images/test',
    #                                jobs_set=edge_to_evaluate)

    # Benchmarking.run_bsds500_boundary_benchmark(input_location='Logs/application_results',
    #                                             gt_location='TestData/BSR/BSDS500/data/groundTruth/test',
    #                                             raw_image='TestData/BSR/BSDS500/data/images/test',
    #                                             jobs_set=edge_to_evaluate, do_thinning=False)

    # Utils.create_latex_fom_table(number_decimal=3, number_of_series=15)
    # print('_______')
    # Utils.create_latex_fom_table(number_decimal=3, number_of_series=15, data='PSNR')

    Utils.close_files()


if __name__ == "__main__":
    # main_sablon()
    main()